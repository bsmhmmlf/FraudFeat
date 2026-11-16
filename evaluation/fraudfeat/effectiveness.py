# -*- coding: utf-8 -*-
"""Evaluate feature effectiveness locally with AutobinSplit information value.

The engine reads released Parquet/CSV data directly and requires no graph service.
The analysis pool is the input-file prefix selected by ``--pool-limit``; only after
that cutoff are valid entries ranked by IV for top-k metrics. Duplicate feature
names remain distinct through ``feature_name#pool_index`` identities.

Evaluation labels contain personal WeChat Pay accounts (fraud=1, non-fraud=0).
``--exclude-targets`` removes discovery targets from the evaluation sample without
removing them from graph traversal. Amount thresholds and aggregates use fen by
default, matching the released edge attributes and EvoFeat; ``--unit yuan``
converts thresholds and aggregates to yuan.

Example::

    python -m fraudfeat.effectiveness --data ./data --features features.jsonl \\
        --out-dir ./results --pool-limit 100 --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .autobin import AutobinSplit
from .graph_store import GraphStore, labels_from_frame, load_schema
from .gremlin_engine import (
    execute_plan_local, is_zero_hit_value, normalize_plan, ratio_den_zero_of,
    validate_plan,
)
from .pool import (
    analysis_pool, entry_key, load_features as _load_feature_file,
    prepare_features,
)


__all__ = ["load_features", "load_eval_set", "evaluate_feature", "evaluate_all",
           "summarize", "render_report", "main"]
logger = logging.getLogger("fraudfeat.effectiveness")
_TOP_K = (10, 30, 50, 100)


def load_features(path: str | Path) -> List[Dict[str, Any]]:
    """Load and normalize a feature submission with stable pool identities."""
    return _load_feature_file(path)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    return pd.read_parquet(path)


def load_eval_set(data_dir: str | Path, *, exclude: Optional[Iterable[str]] = None,
                  limit: int = 0, seed: int = 0,
                  eval_panel: Optional[str | Path] = None) -> Tuple[List[str], Dict[str, int]]:
    data_dir = Path(data_dir)
    if eval_panel is not None and limit:
        raise ValueError("--eval-panel cannot be combined with --limit")
    schema = load_schema(data_dir)
    ext = "csv" if schema.get("format") == "csv" else "parquet"
    labels_file = (schema.get("labels") or {}).get("file") or f"labels.{ext}"
    p = data_dir / labels_file
    if not p.exists():
        raise FileNotFoundError(f"Missing label file for WeChat Pay accounts: {p}")
    df = _read_table(p)
    labels = labels_from_frame(df, p)
    if eval_panel is not None:
        panel_path = Path(eval_panel)
        if not panel_path.is_absolute() and not panel_path.exists():
            candidate = data_dir / panel_path
            if candidate.exists():
                panel_path = candidate
        if panel_path.suffix.lower() not in (".parquet", ".csv"):
            raise ValueError("Evaluation panel must be a Parquet or CSV table")
        panel_labels = labels_from_frame(_read_table(panel_path), panel_path)
        unknown = sorted(set(panel_labels).difference(labels))
        if unknown:
            raise ValueError(
                f"Evaluation panel references {len(unknown)} accounts absent from labels; "
                f"first IDs: {unknown[:3]}")
        ids = list(panel_labels)
    else:
        ids = sorted(labels)
    if exclude:
        drop = {str(x).strip() for x in exclude if str(x).strip()}
        ids = [u for u in ids if u not in drop]
    if eval_panel is None and limit and limit < len(ids):
        rng = np.random.default_rng(seed)
        by_class = {value: [uid for uid in ids if labels[uid] == value] for value in (0, 1)}
        if limit >= 2 and all(by_class.values()):
            selected = [str(rng.choice(by_class[value])) for value in (0, 1)]
            remaining = [uid for uid in ids if uid not in selected]
            if limit > 2:
                selected += rng.choice(np.array(remaining, dtype=object), size=limit - 2,
                                       replace=False).tolist()
            ids = sorted(selected)
        else:
            ids = sorted(rng.choice(np.array(ids, dtype=object), size=limit,
                                    replace=False).tolist())
    selected_labels = {u: labels[u] for u in ids}
    if ids and len(set(selected_labels.values())) < 2:
        raise ValueError("Evaluation set must contain both label 0 and label 1 accounts")
    return ids, selected_labels


def load_targets(path: str | Path) -> List[str]:
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] in "[{":
        obj = json.loads(text)
        items = obj.get("targets") if isinstance(obj, dict) else obj
        return [str(x) for x in (items or [])]
    return [line.strip() for line in text.splitlines() if line.strip()]


def _unique_errors(msgs: List[str], cap: int = 3) -> List[str]:
    seen: "OrderedDict[str, int]" = OrderedDict()
    for m in msgs:
        seen[m] = seen.get(m, 0) + 1
    out = []
    for msg, n in list(seen.items())[:cap]:
        out.append(msg if n == 1 else f"{msg} ({n} occurrences)")
    if len(seen) > cap:
        out.append(f"... and {len(seen) - cap} additional distinct errors")
    return out


def evaluate_feature(store: GraphStore, feat: Dict[str, Any], eval_ids: List[str], *,
                     labels: Dict[str, int], amount_unit: str = "fen",
                     splitter: Optional[AutobinSplit] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "feature_name": feat.get("feature_name") or "",
        "feature_name_cn": feat.get("feature_name_cn") or "",
        "pool_index": feat.get("pool_index"),
        "entry_key": entry_key(feat),
        "window": feat.get("window"),
        "logic": feat.get("logic") or "",
        "expected_value_type": feat.get("expected_value_type") or "",
        "n_seeds": len(eval_ids),
        "valid": False, "exec_ok": False, "non_empty": False,
        "n_ok": 0, "n_error": 0, "n_zero_hit": 0, "coverage": 0.0,
        "iv": None, "n_bins": None, "missing_rate": None,
        "errors": [],
    }
    plan = feat.get("plan") or {}
    if not plan.get("steps"):
        row["errors"] = ["Missing gremlin_plan or empty steps"]
        return row
    norm = normalize_plan(plan)
    ok, msg = validate_plan(norm)
    if not ok:
        row["errors"] = [f"Invalid plan structure: {msg}"]
        return row
    final_type = str((norm.get("final") or {}).get("type") or "")
    values: List[Optional[float]] = []
    errs: List[str] = []
    n_zero_hit = 0
    for uid in eval_ids:
        value, trace, err = execute_plan_local(store, norm, uid, amount_unit=amount_unit)
        if err:
            errs.append(err)
            values.append(None)
            continue
        if is_zero_hit_value(value, final_type=final_type,
                             den_zero=ratio_den_zero_of(trace)):
            n_zero_hit += 1
        try:
            values.append(float(value) if value is not None else None)
        except (TypeError, ValueError):
            values.append(None)
    n_ok = len(eval_ids) - len(errs)
    row["n_ok"] = n_ok
    row["n_error"] = len(errs)
    row["n_zero_hit"] = n_zero_hit
    row["coverage"] = round(n_ok / len(eval_ids), 6) if eval_ids else 0.0
    row["exec_ok"] = not errs
    row["non_empty"] = bool(n_ok and n_zero_hit < n_ok)
    if errs:
        row["errors"] = _unique_errors(errs)
    if not n_ok:
        return row
    df = pd.DataFrame({
        "value": values,
        "label": [int(labels[u]) for u in eval_ids],
    })
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    non_constant = bool(df["value"].dropna().nunique() > 1)
    row["non_constant"] = non_constant
    if not non_constant:
        row["errors"].append("constant feature values are invalid (zero variance)")
    splitter = splitter or AutobinSplit(smoothing_factor=0.5)
    try:
        metrics, _bins = splitter.split_evaluation(df, feature_cols=["value"],
                                                   label_col="label")
    except Exception as e:                                # noqa: BLE001
        row["errors"] = [f"Binning failed: {type(e).__name__}: {e}"]
        return row
    m = metrics.get("value")
    if m is not None:
        iv = float(m.iv)
        if np.isfinite(iv):
            row["iv"] = round(iv, 6)
        else:
            row["errors"] = ["IV is not finite; the evaluation set may contain only one class"]
        row["n_bins"] = int(m.n_bins)
        row["missing_rate"] = round(float(m.missing_rate), 6)
    row["valid"] = bool(row["exec_ok"] and row["non_empty"] and non_constant
                         and row["iv"] is not None)
    return row


def _validate_eval_targets(store: GraphStore, eval_ids: List[str],
                           labels: Dict[str, int]) -> None:
    if eval_ids and len({labels[uid] for uid in eval_ids}) < 2:
        raise ValueError("Evaluation set must contain both label 0 and label 1 accounts")
    unknown = [uid for uid in eval_ids if not store.has_vertex(uid)]
    if unknown:
        raise ValueError(f"Evaluation set references {len(unknown)} unknown account IDs; "
                         f"first={unknown[0]}")


def evaluate_all(store: GraphStore, features: List[Dict[str, Any]],
                 eval_ids: List[str], *, labels: Dict[str, int],
                 amount_unit: str = "fen") -> List[Dict[str, Any]]:
    """Evaluate entries serially in one process."""
    _validate_eval_targets(store, eval_ids, labels)
    features = prepare_features(features)
    splitter = AutobinSplit(smoothing_factor=0.5)
    out = []
    for i, feat in enumerate(features, 1):
        row = evaluate_feature(store, feat, eval_ids, labels=labels,
                               amount_unit=amount_unit, splitter=splitter)
        out.append(row)
        logger.info("[%d/%d] %s iv=%s valid=%s coverage=%.2f",
                    i, len(features), row["feature_name"], row["iv"],
                    row["valid"], row["coverage"])
    return out


_W: Dict[str, Any] = {}


def _init_worker(data_dir: str, eval_ids: List[str], labels: Dict[str, int],
                 amount_unit: str) -> None:
    _W["store"] = GraphStore(data_dir, load_labels=False)
    _validate_eval_targets(_W["store"], eval_ids, labels)
    _W["ids"] = eval_ids
    _W["labels"] = labels
    _W["unit"] = amount_unit
    _W["splitter"] = AutobinSplit(smoothing_factor=0.5)


def _eval_task(feat: Dict[str, Any]) -> Dict[str, Any]:
    return evaluate_feature(_W["store"], feat, _W["ids"], labels=_W["labels"],
                            amount_unit=_W["unit"], splitter=_W["splitter"])


def summarize(rows: List[Dict[str, Any]], *, pool_limit: int = 0) -> Dict[str, Any]:
    """Select the input prefix, then rank valid entries by IV within that pool."""
    all_rows = prepare_features(rows)
    pool = analysis_pool(all_rows, pool_limit)
    valid = [r for r in pool if r.get("valid") and r.get("iv") is not None]
    ranked = sorted(valid, key=lambda r: float(r["iv"]), reverse=True)
    ivs = [float(r["iv"]) for r in ranked]
    summary: Dict[str, Any] = {
        "features_in": len(all_rows),
        "features": len(pool),
        "valid": len(valid),
        "invalid": len(pool) - len(valid),
        "pool_size": len(pool),
        "pool_limit": pool_limit,
        "high_iv_count": sum(1 for value in ivs if value >= 0.5),
        "non_low_iv_count": sum(1 for value in ivs if value >= 0.2),
        "median_iv": round(float(np.median(ivs)), 6) if ivs else None,
        "invalid_reasons": dict(Counter(
            (r.get("errors") or ["(unspecified)"])[0]
            for r in pool if not r.get("valid")
        ).most_common(10)),
    }
    for k in _TOP_K:
        sel = ranked[:k]
        summary[f"top{k}_avg_iv"] = (round(float(np.mean([r["iv"] for r in sel])), 6)
                                     if sel else None)
        summary[f"top{k}_n"] = len(sel)
    summary["max_iv"] = round(float(ranked[0]["iv"]), 6) if ranked else None
    summary["avg_coverage"] = (round(float(np.mean([r["coverage"] for r in pool])), 6)
                              if pool else None)
    summary["top_features"] = [
        {"feature_name": r["feature_name"],
         "pool_index": r.get("pool_index"), "entry_key": entry_key(r),
         "window": r.get("window"), "iv": r["iv"], "coverage": r["coverage"]}
        for r in ranked[:20]
    ]
    return summary


def render_report(rows: List[Dict[str, Any]], summary: Dict[str, Any], *,
                  data_dir: str, features_path: str, eval_size: int,
                  excluded: int, amount_unit: str) -> str:
    lines = [
        "FraudFeat Effectiveness Report (IV)",
        "=" * 60,
        f"Data directory      : {data_dir}",
        f"Features file       : {features_path}",
        f"Evaluation accounts : {eval_size} (excluded discovery targets: {excluded})",
        f"Amount unit         : {amount_unit}",
        f"Features            : {summary['features']} "
        f"(valid={summary['valid']}, invalid={summary['invalid']})",
        f"Analysis pool size  : {summary['pool_size']}",
        f"Pool limit          : {summary['pool_limit'] or 'all'}",
        f"Mean coverage       : {summary['avg_coverage']}",
        "",
        "Top-k mean IV (valid features ranked by descending IV):",
    ]
    for k in _TOP_K:
        v = summary.get(f"top{k}_avg_iv")
        lines.append(f"  top{k:<4}: {v if v is not None else 'N/A'}")
    lines += ["", "Top 20 features:"]
    for i, r in enumerate(summary["top_features"], 1):
        lines.append(f"  {i:>2}. {r['feature_name']:<48} IV={r['iv']:<8} "
                     f"cov={r['coverage']} win={r['window']}")
    if summary["invalid_reasons"]:
        lines += ["", "Invalidity reasons (first error per feature, up to 10 types):"]
        for reason, n in summary["invalid_reasons"].items():
            lines.append(f"  [{n:>3}] {reason}")
    err_rows = [r for r in rows if r.get("errors")]
    if err_rows:
        lines += ["", "Error examples (up to 3 full messages per feature):"]
        for r in err_rows[:20]:
            for e in r["errors"]:
                lines.append(f"  {r['feature_name']}: {e}")
    return "\n".join(lines) + "\n"


# ==================== CLI ====================
def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m fraudfeat.effectiveness",
        description="FraudFeat effectiveness using local plan execution, AutobinSplit, and IV")
    ap.add_argument("--data", default="data", help="Released data directory containing schema.json")
    ap.add_argument("--features", required=True,
                    help="Feature file: features.json or features.jsonl")
    ap.add_argument("--out-dir", default="results", help="Output directory")
    ap.add_argument("--exclude-targets", default=None,
                    help="Discovery-target list; these accounts are excluded from evaluation labels")
    ap.add_argument("--eval-panel", default=None,
                    help="Fixed Parquet/CSV panel with unique_id,label; mutually exclusive with --limit")
    ap.add_argument("--limit", type=int, default=0, help="Evaluation-account sample size; 0 means all")
    ap.add_argument("--seed", type=int, default=0, help="Evaluation-account sampling seed")
    ap.add_argument("--unit", choices=["yuan", "fen"], default="fen",
                    help="Amount unit: fen matches EvoFeat; yuan scales thresholds/aggregates")
    ap.add_argument("--workers", type=int, default=1,
                    help="Reader processes; each builds a full read-only in-memory index")
    ap.add_argument("--pool-limit", type=int, default=100,
                    help="Take the first N features before within-pool IV ranking; 0 means all")
    ap.add_argument("--quiet", action="store_true", help="Suppress the printed report")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    data_dir = Path(args.data)
    all_features = load_features(args.features)
    features = analysis_pool(all_features, args.pool_limit)
    if not features:
        logger.error("No evaluable features found in %s", args.features)
        return 2
    exclude = load_targets(args.exclude_targets) if args.exclude_targets else []
    eval_ids, labels = load_eval_set(data_dir, exclude=exclude,
                                     limit=args.limit, seed=args.seed,
                                     eval_panel=args.eval_panel)
    if not eval_ids:
        logger.error("Evaluation set is empty; check labels and --exclude-targets")
        return 2
    logger.info("Evaluation accounts=%d (excluded targets=%d), analysis pool=%d/%d",
                len(eval_ids), len(exclude), len(features), len(all_features))

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                 initargs=(str(data_dir), eval_ids, labels,
                                           args.unit)) as ex:
            rows = list(ex.map(_eval_task, features))
    else:
        store = GraphStore(data_dir, load_labels=False)
        rows = evaluate_all(store, features, eval_ids, labels=labels,
                            amount_unit=args.unit)
    summary = summarize(rows, pool_limit=args.pool_limit)
    summary["features_in"] = len(all_features)
    summary["eval_size"] = len(eval_ids)
    report = render_report(rows, summary, data_dir=str(data_dir),
                           features_path=str(args.features), eval_size=len(eval_ids),
                           excluded=len(exclude), amount_unit=args.unit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "effectiveness.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "effectiveness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "effectiveness_report.txt").write_text(report, encoding="utf-8")
    if not args.quiet:
        sys.stdout.write(report)
    logger.info("Results written to %s (effectiveness.jsonl, summary JSON, report TXT)", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
