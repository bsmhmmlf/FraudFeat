from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from ._common import parse_line
from .aggregate import _fmt_v, _mean_std, collect, display_method, method_of


_COLS = [
    ("valid_count", "Valid", None),
    ("high_iv_count", "High IV", None),
    ("topk_avg_iv", "k10", ("k", 10)),
    ("topk_avg_iv", "k30", ("k", 30)),
    ("topk_avg_iv", "k50", ("k", 50)),
    ("topk_avg_iv", "k100", ("k", 100)),
    ("distinct_families", "Families", None),
    ("avg_pairwise_sim", "Similarity", None),
    ("near_dup_ratio", "Near dup", None),
    ("truncation_rate", "Trunc rate", None),
]
_SUM = [
    ("valid_count", "Valid feature count", None, "value"),
    ("median_iv", "Median IV", None, "value"),
    ("non_low_iv_count", "Feature count with IV≥0.2", None, "value"),
    ("high_iv_count", "High-IV feature count", None, "value"),
    ("topk_avg_iv", "Top-k mean IV", "k", "value"),
    ("efficiency", "Subgraphs needed to reach k", "k", "subgraphs"),
    ("distinct_families", "Distinct family count", None, "value"),
    ("avg_pairwise_sim", "Mean pairwise similarity", None, "value"),
    ("near_dup_ratio", "Near-duplicate pair ratio", None, "value"),
    ("dup_ratio", "Redundancy rate (duplicate structures)", None, "value"),
    ("topk_distinct_families", "Top-k distinct family count", "k", "value"),
    ("topk_avg_pairwise_sim", "Top-k mean pairwise similarity", "k", "value"),
    ("topk_near_dup_ratio", "Top-k near-duplicate pair ratio", "k", "value"),
    ("embed_avg_cosine", "Semantic embeddings: mean pairwise cosine (higher = more homogeneous)", None, "value"),
    ("embed_nn_distance", "Semantic embeddings: mean nearest-neighbor distance (higher = more diverse)", None, "value"),
    ("topk_embed_avg_cosine", "Top-k embeddings: mean pairwise cosine", "k", "value"),
    ("topk_embed_nn_distance", "Top-k embeddings: mean nearest-neighbor distance", "k", "value"),
    ("truncation_rate", "IV computation truncation rate", None, "value"),
    ("validation_success", "Query validation success rate", None, "value"),
    ("exec_success_rate", "Execution success rate", None, "value"),
    ("judge_score", "Interpretability score (three-judge mean)", None, "value"),
    ("topk_judge_score", "Top-k interpretability score (three-judge mean)", "k", "value"),
    ("judge_w", "Judge agreement W", None, "value"),
    ("judge_coverage_rate", "Scoring coverage rate", None, "value"),
    ("judge_stability", "Judge score stability (variance)", None, "value"),
]
_RATE_KEYS = [("exec_failed", "Exec failed"), ("inconsistent", "Inconsistent"),
              ("truncated", "Truncated"),
              ("effective", "Effective"), ("residual", "Residual")]
_TXT_MODS = (("metrics.txt", "metrics"),
             ("calibration.txt", "evaluator_calibration"),
             ("error_evolution.txt", "error_evolution"), ("cases_index.txt", "cases"))
_RQ2_TXT_MODS = (("iv_curve.txt", "iv_curve"),
                 ("embedding_curve.txt", "embedding_curve"))
_RQ2_METRICS = {
    "evaluator_tier_acc": "inc",
    "evaluator_abs_div": "inc",
    "explorer_pool_families": "cum",
    "explorer_pool_struct_sim": "cum",
    "explorer_pool_near_dup": "cum",
    "embed_pool_avg_cosine": "cum",
    "explorer_pool_top100_iv": "cum",
}


def _kvs_of(rows: list[tuple[str, dict]], metric: str,
            dim=None) -> list[dict]:
    out = []
    for name, kvs in rows:
        if name != metric:
            continue
        if isinstance(dim, tuple):
            if kvs.get(dim[0]) != dim[1]:
                continue
        out.append(kvs)
    return out


def _first(rows, metric, dim=None, value_key: str = "value"):
    got = _kvs_of(rows, metric, dim)
    return (got[0].get(value_key) if got else None)


def _summary_of(run_dir: Path) -> dict:
    p = run_dir / "run_summary.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _collection_of(run_dir: Path) -> str:
    p = run_dir / "run.log"
    if not p.is_file():
        return ""
    m = None
    for m in re.finditer(r"collection=(col-[0-9a-f]+)",
                         p.read_text(encoding="utf-8", errors="replace")):
        pass
    return m.group(1) if m else ""


def _judge_state(run_dir: Path) -> str:
    if (run_dir / "scores.jsonl").is_file() and (run_dir / "scores.jsonl").stat().st_size:
        return "scored"
    if (run_dir / "judge_requests.jsonl").is_file():
        return "pending"
    return "missing"


def _pad(s, w: int) -> str:
    s = str(s)
    width = sum(2 if ord(c) > 0x2E80 else 1 for c in s)
    return s + " " * max(1, w - width)


def _table(header: list[tuple[str, int]], rows: list[list[str]]) -> list[str]:
    out = [" ".join(_pad(h, w) for h, w in header).rstrip()]
    out.append("-" * len(out[0]))
    for r in rows:
        out.append(" ".join(_pad(c, w) for c, (_, w) in zip(r, header)).rstrip())
    return out


def _build_rq2_report(run_dirs: list[Path]) -> list[str]:
    grouped: dict[tuple[str, int, str], list[float]] = {}
    for run_dir in run_dirs:
        for filename in ("iv_curve.txt", "embedding_curve.txt"):
            path = run_dir / filename
            if not path.is_file():
                continue
            for raw in path.read_text(encoding="utf-8").splitlines():
                parsed = parse_line(raw)
                if not parsed:
                    continue
                metric, values = parsed
                episode = values.get("episode")
                value = values.get("value")
                if (_RQ2_METRICS.get(metric) != values.get("span")
                        or not isinstance(episode, int)
                        or not isinstance(value, (int, float))):
                    continue
                grouped.setdefault((metric, episode, values["span"]), []).append(float(value))
    lines = [f"# Ours RQ2: unweighted means over {len(run_dirs)} target streams"]
    for metric in _RQ2_METRICS:
        keys = sorted(key for key in grouped if key[0] == metric)
        for name, episode, span in keys:
            values = grouped[(name, episode, span)]
            lines.append(f"metric={name} episode={episode} span={span} "
                         f"value={sum(values) / len(values):.6f} n_runs={len(values)}")
    return lines


def build_report(method: str, runs_dir: Path, run_dirs: list[Path],
                 pool_limit: int = 0) -> list[str]:
    if method == "ours_rq2":
        return _build_rq2_report(run_dirs)
    data = collect(runs_dir)
    rows_by_run: dict[str, list[tuple[str, dict]]] = {d.name: data.get(d.name, [])
                                                      for d in run_dirs}
    lines = [f"# Method report: {display_method(method)} | {len(run_dirs)} runs | Generated "
             f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"]
    s0 = _summary_of(run_dirs[0]) if run_dirs else {}
    lines.append(f"# Settings: retriever={s0.get('retriever_mode', '-')}"
                 f" collection={_collection_of(run_dirs[0]) or '-'}"
                 f" eval_skill={s0.get('eval_skill_mode', '-')}"
                 + (f" pool_limit={pool_limit}" if pool_limit else "")
                 + " (should match across runs of the same method)")
    lines.append(f"# Seeds: {' '.join(sorted(d.name.split('__s')[-1] for d in run_dirs))}")
    lines.append("")
    lines.append("== 1. Per-run overview ==")
    head = [("run", 20), ("Pool", 5), ("Episodes", 5), ("Subgraphs", 5), ("LLM", 6)]
    head += [(c[1], 7) for c in _COLS] + [("judge", 8)]
    trows = []
    for d in run_dirs:
        s = _summary_of(d)
        row = [d.name, s.get("final_pool_size", "-"), s.get("episodes", "-"),
               s.get("subgraphs_processed", "-"), s.get("llm_calls", "-")]
        for metric, _cn, dim in _COLS:
            v = _first(rows_by_run[d.name], metric, dim)
            row.append(_fmt_v(v) if v is not None else "-")
        js = _first(rows_by_run[d.name], "judge_score")
        row.append(f"{_fmt_v(js)}" if js is not None else _judge_state(d))
        trows.append(row)
    lines += _table(head, trows)
    lines.append("")
    lines.append("== 2. Cross-seed summary (mean±std; value only for a single run) ==")
    sum_head = [("Metric", 26), ("Value", 22), ("n", 4)]
    srows = []
    for metric, label, dim, value_key in _SUM:
        if isinstance(dim, str):
            ks = sorted({kvs.get(dim) for d in run_dirs
                         for kvs in _kvs_of(rows_by_run[d.name], metric, None)
                         if kvs.get(dim) is not None})
            for k in ks:
                vals = []
                for d in run_dirs:
                    v = _first(rows_by_run[d.name], metric, (dim, k), value_key)
                    if v is not None:
                        vals.append(v)
                if vals:
                    srows.append([f"{label} ({dim}={k})", _mean_std([float(v) for v in vals]),
                                  len(vals)])
        else:
            vals = []
            for d in run_dirs:
                v = _first(rows_by_run[d.name], metric, None, value_key)
                if v is not None:
                    vals.append(v)
            if vals:
                srows.append([label, _mean_std([float(v) for v in vals]), len(vals)])
    lines += _table(sum_head, srows)
    lines.append("")
    lines.append("== 3. Five-rate funnel / cost (Appendix D) ==")
    lines.append("(Rates stop at analysis cutoff episode W*; costs cover the full run; per-feature cost = full-run cost / full-run valid features)")
    r_head = [("run", 20), ("W*", 4)] + [(cn, 8) for _k, cn in _RATE_KEYS] + [
        ("LLM calls", 8), ("tokens", 12), ("Graph queries", 8), ("Calls/feature", 9), ("Tok/feature", 10)]
    rrows = []
    for d in run_dirs:
        rr = rows_by_run[d.name]
        rate = _kvs_of(rr, "rate")
        cost = _kvs_of(rr, "cost")
        c0 = cost[0] if cost else {}
        cpv = _kvs_of(rr, "cost_per_valid")
        p0 = cpv[0] if cpv else {}
        tk = None
        if c0.get("prompt_tokens") is not None or c0.get("completion_tokens") is not None:
            tk = (int(c0.get("prompt_tokens") or 0) + int(c0.get("completion_tokens") or 0))
        tkpv = None
        if p0.get("prompt_tokens") is not None or p0.get("completion_tokens") is not None:
            tkpv = round(float(p0.get("prompt_tokens") or 0)
                         + float(p0.get("completion_tokens") or 0), 1)
        rrows.append([d.name, _fmt_v(_first(rr, "analysis_cutoff", None, "episode"))]
                     + [_fmt_v(rate[0].get(k)) if rate else "-"
                        for k, _cn in _RATE_KEYS]
                     + [_fmt_v(c0.get("llm_calls")), _fmt_v(tk),
                        _fmt_v(c0.get("graph_queries")),
                        _fmt_v(p0.get("llm_calls")), _fmt_v(tkpv)])
    lines += _table(r_head, rrows)
    lines.append("")
    lines.append("== 4. Judges (metric 3: interpretability | analysis set × three judges, limited by --pool-limit) ==")
    j_head = [("run", 20), ("Status", 8), ("Score", 8), ("Coverage", 8), ("W", 8), ("Stability", 8)]
    jrows = []
    for d in run_dirs:
        rr = rows_by_run[d.name]
        jrows.append([d.name, _judge_state(d),
                      _fmt_v(_first(rr, "judge_score")),
                      _fmt_v(_first(rr, "judge_coverage_rate")),
                      _fmt_v(_first(rr, "judge_w")),
                      _fmt_v(_first(rr, "judge_stability"))])
    lines += _table(j_head, jrows)
    tk_head = [("run", 20), ("k10", 8), ("k30", 8), ("k50", 8), ("k100", 8)]
    tkrows = []
    for d in run_dirs:
        rr = rows_by_run[d.name]
        cells = []
        for k in (10, 30, 50, 100):
            got = _kvs_of(rr, "topk_judge_score", ("k", k))
            cells.append(_fmt_v(got[0].get("value")) if got else "-")
        tkrows.append([d.name] + cells)
    lines += _table(tk_head, tkrows)
    states = {_judge_state(d) for d in run_dirs}
    if "pending" in states or "missing" in states:
        lines.append("")
        lines.append("Judge scoring is incomplete; complete it with the unified evaluation entry point:")
        lines.append("   bash run_evaluation.sh")
    lines.append("")
    lines.append("== 5. Raw metric lines (copyable / parseable) ==")
    for d in run_dirs:
        lines.append(f"-- run={d.name} ({len(rows_by_run[d.name])} lines)")
        for name, kvs in rows_by_run[d.name]:
            kv = " ".join(f"{k}={_fmt_v(v)}" for k, v in kvs.items())
            lines.append(f"metric={name} {kv}".rstrip())
    return lines


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m experiments.report")
    ap.add_argument("method", help="Paper method name (e.g. ours / OURS / all)")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--gen", action="store_true",
                    help="Generate missing metric txt files (metrics/iv_curve/calibration/"
                         "error_evolution/cases; excludes diversity_embedding)")
    ap.add_argument("--pool-limit", type=int, default=0,
                    help="Metrics 1 and 2 use the first N features.json entries in file order "
                         "(final paper setting: 100); 0=full pool. When >0, --gen reruns metrics "
                         "to match the analysis scope; other metric txt files are generated if missing")
    ap.add_argument("--stdout", action="store_true", help="Also print the report to stdout")
    args = ap.parse_args(argv)
    runs_dir = Path(args.runs)
    if not runs_dir.is_dir():
        raise SystemExit(f"[report] Runs root does not exist: {runs_dir}")
    method = args.method.strip().lower()
    all_methods = sorted({method_of(d.name) for d in runs_dir.iterdir()
                          if d.is_dir() and "__s" in d.name})
    methods = all_methods if method == "all" else [method]
    for m in methods:
        run_dirs = sorted(d for d in runs_dir.iterdir()
                          if d.is_dir() and method_of(d.name) == m)
        if not run_dirs:
            print(f"[report][skip] {m}: no runs under {runs_dir}"
                  f" (available methods: {', '.join(all_methods)})")
            continue
        if args.gen:
            modules = _RQ2_TXT_MODS if m == "ours_rq2" else _TXT_MODS
            for d in run_dirs:
                for fname, mod in modules:
                    force = (m == "ours_rq2"
                             or (mod in ("metrics", "evaluator_calibration")
                                 and args.pool_limit > 0))
                    if (d / fname).is_file() and not force:
                        continue
                    cmd = [sys.executable, "-X", "utf8", "-m",
                           f"experiments.{mod}", str(d)]
                    if mod in ("metrics", "evaluator_calibration") and args.pool_limit > 0:
                        cmd += ["--pool-limit", str(args.pool_limit)]
                    subprocess.run(cmd, check=False)
        lines = build_report(m, runs_dir, run_dirs, pool_limit=args.pool_limit)
        out = runs_dir / f"report_{m}.txt"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[report] {m}: {len(run_dirs)} runs → {out} ({len(lines)} lines)")
        if args.stdout:
            print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
