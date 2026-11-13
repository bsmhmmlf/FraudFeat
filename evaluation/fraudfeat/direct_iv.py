#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute IV directly from released relation tables without Gremlin plans."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .autobin import AutobinSplit
from .graph_store import labels_from_frame, load_schema


logger = logging.getLogger("fraudfeat.direct_iv")
DEFAULT_METRICS = (
    "degree", "edge_count", "trade_count_sum",
    "amount_sum", "amount_mean", "amount_min", "amount_max",
)
METRIC_COLUMNS = {
    "trade_count_sum": "trade_count",
    "amount_sum": "total_amount",
    "amount_mean": "total_amount",
    "amount_min": "total_amount",
    "amount_max": "total_amount",
    "top_k_min": "top_k",
}
ALLOWED_METRICS = frozenset(("degree", "edge_count", "top_k_min", *METRIC_COLUMNS))


def _read_table(path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required table is missing: {path}")
    if path.suffix.lower() == ".csv":
        if columns is None:
            return pd.read_csv(path)
        available = list(pd.read_csv(path, nrows=0).columns)
        selected = [name for name in columns if name in available]
        return pd.read_csv(path, usecols=selected)
    import pyarrow.parquet as pq
    if columns is None:
        return pd.read_parquet(path)
    available = list(pq.ParquetFile(str(path)).schema_arrow.names)
    selected = [name for name in columns if name in available]
    return pd.read_parquet(path, columns=selected)


def _load_targets(path: str | Path) -> List[str]:
    target_path = Path(path)
    text = target_path.read_text(encoding="utf-8")
    if target_path.suffix.lower() == ".jsonl":
        values = []
        for line in text.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            values.append(str(item.get("unique_id") or item.get("id") or item.get("target_id") or "")
                          if isinstance(item, dict) else str(item))
        return [value for value in values if value]
    if target_path.suffix.lower() == ".json":
        item = json.loads(text)
        values = item.get("targets") if isinstance(item, dict) else item
        return [str(value) for value in (values or [])]
    return [line.strip() for line in text.splitlines() if line.strip()]


def _load_eval_set(data_dir: Path, *, exclude: Sequence[str], limit: int,
                   seed: int, eval_panel: Optional[str | Path] = None
                   ) -> Tuple[List[str], Dict[str, int]]:
    if eval_panel is not None and limit:
        raise ValueError("--eval-panel cannot be combined with --limit")
    schema = load_schema(data_dir)
    path = data_dir / str(schema["labels"]["file"])
    labels = labels_from_frame(_read_table(path), path)
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
                f"Evaluation panel references {len(unknown)} accounts absent from labels")
        ids = list(panel_labels)
    else:
        ids = sorted(labels)
    dropped = {str(value).strip() for value in exclude if str(value).strip()}
    ids = [uid for uid in ids if uid not in dropped]
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
    selected_labels = {uid: labels[uid] for uid in ids}
    if ids and len(set(selected_labels.values())) < 2:
        raise ValueError("Evaluation set must contain both label 0 and label 1 accounts")
    return ids, selected_labels


def _validate_scope_accounts(data_dir: Path, schema: Dict[str, Any], scope: str,
                             eval_ids: List[str]) -> None:
    entity = next((item for item in schema.get("entities") or []
                   if item.get("entity_type") == scope), None)
    if entity is None:
        raise ValueError(f"Label scope {scope} is not declared as an entity")
    path = data_dir / str(entity["file"])
    frame = _read_table(path, ["unique_id"])
    if "unique_id" not in frame.columns:
        raise ValueError(f"Scope node file {path} lacks unique_id")
    if frame["unique_id"].duplicated().any():
        raise ValueError(f"Scope node file {path} contains duplicate unique_id values")
    known = set(frame["unique_id"].astype(str))
    unknown = [uid for uid in eval_ids if uid not in known]
    if unknown:
        raise ValueError(f"Labels reference {len(unknown)} unknown scope accounts; first={unknown[0]}")


def _summarize_direct(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if row.get("valid") and row.get("iv") is not None]
    ranked = sorted(valid, key=lambda row: float(row["iv"]), reverse=True)
    iv_values = [float(row["iv"]) for row in valid]
    summary: Dict[str, Any] = {
        "features": len(rows),
        "valid": len(valid),
        "invalid": len(rows) - len(valid),
        "median_iv": round(float(np.median(iv_values)), 6) if iv_values else None,
        "avg_coverage": round(float(np.mean([row["coverage"] for row in rows])), 6) if rows else None,
        "pool_size": len(rows),
        "pool_limit": 0,
        "top_features": [
            {"feature_name": row["feature_name"], "iv": row["iv"],
             "coverage": row["coverage"], "entry_key": row["entry_key"]}
            for row in ranked[:20]
        ],
    }
    for k in (10, 30, 50, 100):
        subset = ranked[:k]
        summary[f"top{k}_avg_iv"] = (
            round(float(np.mean([row["iv"] for row in subset])), 6) if subset else None)
    return summary


def _selected_relations(schema: Dict[str, Any], requested: Optional[str]) -> List[Dict[str, Any]]:
    relations = list(schema.get("relations") or [])
    if not requested:
        return relations
    wanted = [item.strip() for item in requested.split(",") if item.strip()]
    known = {str(relation.get("en")) for relation in relations}
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise ValueError(f"Unknown public relation names: {unknown}")
    by_name = {str(relation.get("en")): relation for relation in relations}
    return [by_name[name] for name in wanted]


def _relation_endpoints(relation: Dict[str, Any], scope: str) -> Tuple[str, str, List[str]]:
    endpoints = list(relation.get("join_fields") or [])
    if len(endpoints) != 1:
        raise ValueError(f"Relation {relation.get('en')} must declare exactly one src/dst endpoint pair")
    endpoint = endpoints[0]
    src_type, dst_type = str(endpoint.get("src") or ""), str(endpoint.get("dst") or "")
    roles = [side for side, entity in (("src", src_type), ("dst", dst_type)) if entity == scope]
    return src_type, dst_type, roles


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _role_values(frame: pd.DataFrame, role: str, eval_ids: List[str],
                 metrics: Sequence[str]) -> Dict[str, pd.Series]:
    anchor = role
    counterparty = "dst" if role == "src" else "src"
    selected = set(eval_ids)
    subset = frame.loc[frame[anchor].astype(str).isin(selected)].copy()
    subset[anchor] = subset[anchor].astype(str)
    grouped = subset.groupby(anchor, sort=False)
    values: Dict[str, pd.Series] = {}
    for metric in metrics:
        if metric == "degree":
            series = grouped[counterparty].nunique() if len(subset) else pd.Series(dtype=float)
        elif metric == "edge_count":
            series = grouped.size() if len(subset) else pd.Series(dtype=float)
        else:
            column = METRIC_COLUMNS[metric]
            if column not in subset.columns:
                continue
            numeric = _numeric(subset, column)
            subset[f"__{metric}"] = numeric
            operation = {
                "trade_count_sum": "sum", "amount_sum": "sum", "amount_mean": "mean",
                "amount_min": "min", "amount_max": "max", "top_k_min": "min",
            }[metric]
            series = getattr(grouped[f"__{metric}"], operation)() if len(subset) else pd.Series(dtype=float)
        values[metric] = series.reindex(eval_ids).fillna(0.0).astype(float)
    return values


def _expected_type(metric: str) -> str:
    if metric in {"amount_sum", "amount_mean", "amount_min", "amount_max"}:
        return "amount"
    if metric in {"top_k_min"}:
        return "rank"
    return "count"


def evaluate_relation(data_dir: Path, relation: Dict[str, Any], eval_ids: List[str],
                      labels: Dict[str, int], scope: str, metrics: Sequence[str],
                      splitter: AutobinSplit, start_index: int) -> List[Dict[str, Any]]:
    name = str(relation.get("en") or "")
    _src_type, _dst_type, roles = _relation_endpoints(relation, scope)
    if not roles:
        logger.warning("Skipping %s because neither endpoint has label scope %s", name, scope)
        return []
    path = data_dir / str(relation["file"])
    columns = ["src", "dst", "relation", "total_amount", "trade_count", "top_k"]
    frame = _read_table(path, columns)
    if not {"src", "dst", "relation"}.issubset(frame.columns):
        raise ValueError(f"Relation file {path} lacks src/dst/relation columns")
    if not frame.empty and not frame["relation"].astype(str).eq(name).all():
        raise ValueError(f"Relation file {path} contains values outside {name}")

    rows: List[Dict[str, Any]] = []
    for role in roles:
        values = _role_values(frame, role, eval_ids, metrics)
        if not values:
            logger.warning("No requested metrics are available for %s role=%s", name, role)
            continue
        feature_names = [f"{name}__{role}__{metric}" for metric in values]
        frame_values = pd.DataFrame({feature: values[metric].to_numpy()
                                     for feature, metric in zip(feature_names, values)})
        frame_values["label"] = [int(labels[uid]) for uid in eval_ids]
        computed, _bins = splitter.split_evaluation(
            frame_values, feature_cols=feature_names, label_col="label")
        for offset, (feature, metric) in enumerate(zip(feature_names, values)):
            array = values[metric].to_numpy()
            metric_result = computed.get(feature)
            iv = float(metric_result.iv) if metric_result is not None else None
            non_constant = bool(np.unique(array).size > 1)
            non_empty = bool(np.count_nonzero(array) > 0)
            errors = []
            if not non_constant:
                errors.append("constant feature values are invalid (zero variance)")
            if iv is not None and not np.isfinite(iv):
                errors.append("IV is not finite")
                iv = None
            row = {
                "feature_name": feature,
                "pool_index": start_index + len(rows),
                "entry_key": f"{feature}#{start_index + len(rows)}",
                "relation": name,
                "role": role,
                "metric": metric,
                "logic": f"Direct {metric} aggregation over the {role} endpoint of {name}",
                "expected_value_type": _expected_type(metric),
                "method": "direct_table_aggregation",
                "n_seeds": len(eval_ids),
                "valid": bool(non_constant and non_empty and iv is not None),
                "exec_ok": True,
                "non_empty": non_empty,
                "non_constant": non_constant,
                "n_ok": len(eval_ids),
                "n_error": 0,
                "n_zero_hit": int(np.count_nonzero(array == 0)),
                "coverage": 1.0,
                "iv": round(iv, 6) if iv is not None else None,
                "n_bins": int(metric_result.n_bins) if metric_result is not None else None,
                "missing_rate": (round(float(metric_result.missing_rate), 6)
                                 if metric_result is not None else None),
                "errors": errors,
            }
            rows.append(row)
    return rows


def evaluate_direct(data_dir: Path, eval_ids: List[str], labels: Dict[str, int], *,
                    relations: Sequence[Dict[str, Any]], metrics: Sequence[str],
                    scope: str) -> List[Dict[str, Any]]:
    splitter = AutobinSplit(smoothing_factor=0.5)
    rows: List[Dict[str, Any]] = []
    for relation in relations:
        rows.extend(evaluate_relation(data_dir, relation, eval_ids, labels, scope,
                                      metrics, splitter, len(rows)))
    for index, row in enumerate(rows):
        row["pool_index"] = index
        row["entry_key"] = f"{row['feature_name']}#{index}"
    return rows


def render_direct_report(rows: List[Dict[str, Any]], summary: Dict[str, Any], *,
                         data_dir: Path, eval_size: int) -> str:
    lines = [
        "FraudFeat Direct Table IV Report (no Gremlin execution)",
        "=" * 60,
        f"Data directory      : {data_dir}",
        f"Evaluation accounts : {eval_size}",
        f"Features            : {len(rows)} (valid={summary['valid']}, invalid={summary['invalid']})",
        f"Median IV           : {summary.get('median_iv')}",
        "",
        "Top 20 direct features:",
    ]
    for index, row in enumerate(summary.get("top_features") or [], 1):
        lines.append(f"  {index:>2}. {row['feature_name']:<90} IV={row['iv']}")
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fraudfeat.direct_iv",
        description="Compute IV directly from released Parquet/CSV relations without Gremlin")
    parser.add_argument("--data", default="data", help="Released data directory containing schema.json")
    parser.add_argument("--relations", help="Comma-separated public relation names; default is all")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                        help=f"Comma-separated metrics from {sorted(ALLOWED_METRICS)}")
    parser.add_argument("--out-dir", default="results/direct_iv")
    parser.add_argument("--exclude-targets", help="Discovery-target list excluded from evaluation labels")
    parser.add_argument("--eval-panel", help="Fixed Parquet/CSV panel with unique_id,label; excludes --limit")
    parser.add_argument("--limit", type=int, default=0, help="Evaluation-account sample size; 0 means all")
    parser.add_argument("--seed", type=int, default=0, help="Evaluation-account sampling seed")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    metrics = tuple(item.strip() for item in args.metrics.split(",") if item.strip())
    unknown_metrics = [metric for metric in metrics if metric not in ALLOWED_METRICS]
    if unknown_metrics:
        logger.error("Unknown direct metrics: %s", unknown_metrics)
        return 2
    if not metrics:
        logger.error("At least one direct metric is required")
        return 2
    data_dir = Path(args.data)
    try:
        schema = load_schema(data_dir)
        scope = str((schema.get("labels") or {}).get("scope") or "wechat_pay_account")
        relations = _selected_relations(schema, args.relations)
        exclude = _load_targets(args.exclude_targets) if args.exclude_targets else []
        eval_ids, labels = _load_eval_set(data_dir, exclude=exclude,
                                          limit=args.limit, seed=args.seed,
                                          eval_panel=args.eval_panel)
        if not eval_ids:
            logger.error("Evaluation set is empty")
            return 2
        _validate_scope_accounts(data_dir, schema, scope, eval_ids)
        rows = evaluate_direct(data_dir, eval_ids, labels, relations=relations,
                               metrics=metrics, scope=scope)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        logger.error("%s", exc)
        return 2
    if not rows:
        logger.error("No direct features could be computed")
        return 2

    summary = _summarize_direct(rows)
    summary["features_in"] = len(rows)
    summary["eval_size"] = len(eval_ids)
    summary["method"] = "direct_table_aggregation"
    summary["metrics"] = list(metrics)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "direct_iv.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "direct_iv_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render_direct_report(rows, summary, data_dir=data_dir, eval_size=len(eval_ids))
    (out_dir / "direct_iv_report.txt").write_text(report, encoding="utf-8")
    if not args.quiet:
        sys.stdout.write(report)
    logger.info("Direct IV results written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
