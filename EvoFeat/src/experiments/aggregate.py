from __future__ import annotations

import sys
from pathlib import Path

from ._common import parse_line


_MAIN_METRICS = [
    ("valid_count", "Valid feature count"),
    ("median_iv", "Median IV"),
    ("non_low_iv_count", "Feature count with IV≥0.2"),
    ("high_iv_count", "High-IV feature count"),
    ("topk_avg_iv", "Top-k mean IV"),
    ("efficiency", "Subgraphs needed to reach k"),
    ("distinct_families", "Distinct family count"),
    ("avg_pairwise_sim", "Mean pairwise similarity"),
    ("near_dup_ratio", "Near-duplicate pair ratio"),
    ("dup_ratio", "Redundancy rate (duplicate structures)"),
    ("topk_distinct_families", "Top-k distinct family count"),
    ("topk_avg_pairwise_sim", "Top-k mean pairwise similarity"),
    ("topk_near_dup_ratio", "Top-k near-duplicate pair ratio"),
    ("embed_avg_cosine", "Semantic embeddings: mean pairwise cosine"),
    ("embed_nn_distance", "Semantic embeddings: mean nearest-neighbor distance"),
    ("topk_embed_avg_cosine", "Top-k embeddings: mean pairwise cosine"),
    ("topk_embed_nn_distance", "Top-k embeddings: mean nearest-neighbor distance"),
    ("rate", "Five rates (execution failure/inconsistency/truncation/effective)"),
    ("truncation_rate", "IV computation truncation rate"),
    ("validation_success", "Query validation success rate"),
    ("exec_success_rate", "Execution success rate"),
    ("cost", "Cost (LLM calls/tokens/graph queries)"),
    ("cost_per_valid", "Cost per valid feature (full run)"),
    ("analysis_cutoff", "Analysis cutoff episode W* / candidates and pooled features through cutoff"),
    ("judge_score", "Interpretability score (three-judge mean)"),
    ("topk_judge_score", "Top-k interpretability score"),
    ("judge_w", "Judge agreement W"),
    ("judge_coverage_rate", "Scoring coverage rate"),
    ("judge_stability", "Judge score stability (variance)"),
]
_VALUE_ONLY: set = set()
_GROUP_KEYS = ("k",)
_VALUE_KEYS = ("value", "subgraphs")


def collect(runs_dir: Path) -> dict[str, list[tuple[str, dict]]]:
    out: dict[str, list[tuple[str, dict]]] = {}
    for d in sorted(runs_dir.iterdir() if runs_dir.is_dir() else []):
        if not d.is_dir():
            continue
        rows: list[tuple[str, dict]] = []
        for txt in ("metrics.txt", "iv_curve.txt", "error_evolution.txt",
                    "calibration.txt", "judge_report.txt", "judge_scores.txt",
                    "embedding_diversity.txt", "embedding_curve.txt"):
            p = d / txt
            if not p.is_file():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                parsed = parse_line(line)
                if parsed:
                    rows.append(parsed)
        if rows:
            out[d.name] = rows
    return out


def _fmt_v(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


DISPLAY_METHODS = {
    "ours": "Ours", "ours_rq2": "Ours (RQ2)", "manual": "Manual",
    "stat": "Stat", "comm": "Comm", "base_agent": "Base-Agent",
    "static_agent": "Static-Agent", "wo_exp": "w/o-Exp", "wo_syn": "w/o-Syn",
    "wo_eval": "w/o-Eval", "wo_cov": "w/o-Cov",
}


def method_of(run_id: str) -> str:
    return run_id.split("__s")[0]


def display_method(method: str) -> str:
    return DISPLAY_METHODS.get(method, method)


def _mean_std(vals: list[float]) -> str:
    if not vals:
        return "-"
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return _fmt_v(round(m, 4))
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return f"{_fmt_v(round(m, 4))}±{var ** 0.5:.4f}"


def render_table(data: dict[str, list[tuple[str, dict]]]) -> list[str]:
    methods = sorted({method_of(r) for r in data})
    grouped: dict[str, list[str]] = {m: sorted(r for r in data
                                             if method_of(r) == m)
                                     for m in methods}
    header = ["metric".ljust(22), "dim".ljust(14)] + [display_method(m)[:18].ljust(18)
                                                      for m in methods]
    lines = [" ".join(header)]
    lines.append("-" * min(len(lines[0]), 120))
    for metric, _cn in _MAIN_METRICS:
        rows_all = [kvs for r in data for name, kvs in data[r] if name == metric]
        if not rows_all:
            continue
        group_key = next((g for g in _GROUP_KEYS
                          if any(g in kvs for kvs in rows_all)), None)
        value_key = next((v for v in _VALUE_KEYS
                          if any(v in kvs for kvs in rows_all)), "value")
        if group_key:
            for gv in sorted({kvs[group_key] for kvs in rows_all
                              if group_key in kvs}):
                row = [metric.ljust(22), f"{group_key}={gv}".ljust(14)]
                for m in methods:
                    vals = []
                    for r in grouped[m]:
                        for name, kvs in data[r]:
                            if name != metric or kvs.get(group_key) != gv:
                                continue
                            v = kvs.get(value_key)
                            if isinstance(v, (int, float)):
                                vals.append(float(v))
                    row.append((_mean_std(vals) if vals else "-")[:18].ljust(18))
                lines.append(" ".join(row).rstrip())
            continue
        dims: list[str] = [""]
        if metric not in _VALUE_ONLY:
            for kvs in rows_all:
                for k in kvs:
                    if k != value_key and k not in dims:
                        dims.append(k)
        for dim in dims:
            row = [metric.ljust(22), dim.ljust(14)]
            for m in methods:
                vals = []
                for r in grouped[m]:
                    for name, kvs in data[r]:
                        if name != metric or (dim and dim not in kvs):
                            continue
                        v = kvs.get(value_key) if not dim else kvs.get(dim)
                        if isinstance(v, (int, float)):
                            vals.append(float(v))
                row.append((_mean_std(vals) if vals else "-")[:18].ljust(18))
            lines.append(" ".join(row).rstrip())
    return lines


def render_raw(data: dict[str, list[tuple[str, dict]]]) -> list[str]:
    lines = ["", "== Raw metric lines (grouped by run) =="]
    for r, rows in data.items():
        lines.append(f"-- run={r} ({len(rows)} lines)")
        for name, kvs in rows:
            kv = " ".join(f"{k}={_fmt_v(v)}" for k, v in kvs.items())
            lines.append(f"metric={name} {kv}".rstrip())
    return lines


def main(argv: list[str]) -> None:
    runs_dir = Path(argv[0] if argv else "runs")
    data = collect(runs_dir)
    if not data:
        print(f"[aggregate] No metric outputs under {runs_dir}")
        return
    lines = render_table(data) + render_raw(data)
    out = runs_dir / "aggregate.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[aggregate] {len(data)} runs → {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main(sys.argv[1:])
