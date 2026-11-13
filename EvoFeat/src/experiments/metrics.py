from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

from ._common import (analysis_cutoff, effective_iv, emit, parse_line,
                      rank_by_iv, read_candidate_rows, read_history, read_pool,
                      read_usage, reconcile_outcomes, within_cutoff, write_txt)


TOPK = (10, 30, 50, 100)
EFFICIENCY_K = (10, 30, 50, 100)
HIGH_IV = 0.5
_WORKERS: int | None = None
_REUSE_DIV: bool = False


_DIV_NAMES = {"distinct_families", "avg_pairwise_sim", "near_dup_ratio",
              "equivalence_classes", "redundancy_clusters", "max_cluster_size",
              "dup_ratio"}


def _is_div_line(name: str) -> bool:
    base = name[len("topk_"):] if name.startswith("topk_") else name
    return base in _DIV_NAMES


def _reusable_div_lines(run_dir: str | Path, n_pool: int) -> list[str] | None:
    p = Path(run_dir) / "metrics.txt"
    if not p.is_file():
        return None
    old = p.read_text(encoding="utf-8").splitlines()
    valid_old = None
    div: list[str] = []
    for ln in old:
        parsed = parse_line(ln)
        if not parsed:
            continue
        name, kvs = parsed
        if name == "valid_count":
            valid_old = kvs.get("value")
        elif _is_div_line(name):
            div.append(ln.strip())
    if valid_old != n_pool or not any(ln.startswith("metric=distinct_families ")
                                      for ln in div):
        return None
    return div


def _div(pool: list[dict]) -> dict:
    from .structural_diversity import batch_metrics
    return batch_metrics(pool, workers=_WORKERS)


def _emit_div(lines: list[str], bm: dict, prefix: str = "", **extra) -> None:
    lines.append(emit(f"{prefix}distinct_families", value=bm["distinct_structures"], **extra))
    lines.append(emit(f"{prefix}avg_pairwise_sim", value=bm["avg_pairwise_sim"],
                      n_pairs=bm["n_pairs"], **extra))
    lines.append(emit(f"{prefix}near_dup_ratio", value=bm["near_dup_ratio"], **extra))
    lines.append(emit(f"{prefix}equivalence_classes", value=bm["equivalence_classes"], **extra))
    lines.append(emit(f"{prefix}redundancy_clusters", value=bm["redundancy_clusters"], **extra))
    lines.append(emit(f"{prefix}max_cluster_size", value=bm["max_cluster_size"], **extra))
    n = int(bm.get("n") or 0)
    lines.append(emit(f"{prefix}dup_ratio",
                      value=(round(1 - bm["distinct_structures"] / n, 4) if n else None),
                      **extra))


def compute_metrics(run_dir: str | Path, pool_limit: int = 0) -> list[str]:
    lines: list[str] = []
    pool_all = read_pool(run_dir)
    pool = pool_all[:pool_limit] if pool_limit and pool_limit > 0 else pool_all
    cutoff = analysis_cutoff(pool_all, pool_limit)
    pool_upto = [f for f in pool_all if within_cutoff(f, cutoff)]
    rows_all = [r for r in read_candidate_rows(run_dir) if within_cutoff(r, cutoff)]
    cands, unmatched_pool = reconcile_outcomes(rows_all, pool_upto)
    history = read_history(run_dir)
    usage = read_usage(run_dir)
    lines.append(emit("analysis_cutoff", episode=cutoff, pool_limit=pool_limit,
                      pool_head=len(pool), pool_upto=len(pool_upto),
                      pool_total=len(pool_all), produced_upto=len(cands),
                      unmatched_pool=unmatched_pool))
    ivs = sorted((float(f["eval_iv"]) for f in pool
                  if f.get("eval_iv") is not None), reverse=True)
    lines.append(emit("valid_count", value=len(pool)))
    lines.append(emit("high_iv_count", value=sum(1 for v in ivs if v >= HIGH_IV)))
    lines.append(emit("non_low_iv_count", value=sum(1 for v in ivs if v >= 0.2)))
    lines.append(emit("median_iv", value=(round(statistics.median(ivs), 4) if ivs else None),
                      n=len(ivs)))
    avg_ivs = sorted((effective_iv(f["eval_iv"], run_dir) for f in pool
                      if f.get("eval_iv") is not None), reverse=True)
    for k in TOPK:
        top = avg_ivs[:k]
        lines.append(emit("topk_avg_iv", k=k,
                          value=(round(sum(top) / len(top), 4) if top else None),
                          n=len(top)))
    for h in history:
        if h.get("cum_valid") is not None:
            lines.append(emit("efficiency_curve", episode=h.get("episode"),
                              subgraphs=h.get("cum_subgraphs"),
                              cum_valid=h.get("cum_valid")))
    for k in EFFICIENCY_K:
        hit = next((h for h in history
                    if (h.get("cum_valid") or 0) >= k), None)
        lines.append(emit("efficiency", k=k,
                          subgraphs=(hit.get("cum_subgraphs") if hit else None),
                          reached=bool(hit)))
    produced = len(cands)

    def _stage_of(c: dict) -> str:
        s = c.get("death_stage")
        if s:
            return str(s)
        err = str(c.get("exec_error") or "")
        if "consistency=" in err:
            return "consistency"
        return "exec" if err else ""

    def _rate(n: int):
        return round(n / produced, 4) if produced else None

    dead = [c for c in cands if c.get("outcome") == "synth_failed"]
    n_exec = sum(1 for c in dead if _stage_of(c) in ("struct", "exec"))
    n_semantic = sum(1 for c in dead if _stage_of(c) == "semantic")
    n_incons = sum(1 for c in dead
                   if _stage_of(c) in ("consistency", "semantic"))
    n_trunc = sum(1 for c in cands if c.get("outcome") in ("truncated", "pruned_low"))
    n_pool = sum(1 for c in cands if c.get("outcome") == "pooled")
    n_resid = produced - n_exec - n_incons - n_trunc - n_pool
    lines.append(emit("funnel", produced=produced, exec_failed=n_exec,
                      inconsistent=n_incons, semantic=n_semantic,
                      truncated=n_trunc,
                      effective=n_pool, residual=n_resid))
    lines.append(emit("rate", exec_failed=_rate(n_exec), inconsistent=_rate(n_incons),
                      truncated=_rate(n_trunc),
                      effective=_rate(n_pool), residual=_rate(n_resid)))
    passed_gates = [c for c in cands
                    if c.get("outcome") != "synth_failed"]
    accounted = [c for c in passed_gates
                 if c.get("outcome") in ("pooled", "frozen")]
    trunc = (round(1 - len(accounted) / len(passed_gates), 4)
             if passed_gates else None)
    lines.append(emit("truncation_rate", value=trunc,
                      accounted=len(accounted), passed_gates=len(passed_gates)))
    survived_synth = sum(1 for c in cands if c.get("outcome") != "synth_failed")
    lines.append(emit("validation_success",
                      value=(round(survived_synth / produced, 4) if produced else None),
                      passed=survived_synth, n=produced))
    lines.append(emit("survival", stage="produced", count=produced, denom=produced))
    lines.append(emit("survival", stage="synth_passed", count=survived_synth,
                      denom=produced))
    lines.append(emit("survival", stage="accounted", count=len(accounted),
                      denom=produced))
    lines.append(emit("survival", stage="pooled", count=n_pool, denom=produced))
    execd = [c for c in cands if (c.get("exec_attempts") or 0) > 0]
    if execd:
        ok = sum(1 for c in execd if not c.get("exec_error"))
        lines.append(emit("exec_success_rate", value=round(ok / len(execd), 4),
                          n=len(execd)))
    reused = _reusable_div_lines(run_dir, len(pool)) if _REUSE_DIV else None
    if reused is not None:
        lines.extend(reused)
        print(f"[metrics] {run_dir}: reusing {len(reused)} structural diversity lines from metrics.txt")
    else:
        if _REUSE_DIV:
            print(f"[metrics] {run_dir}: cached structural diversity cannot be reused (missing or count mismatch); recomputing GED")
        bm_pool = _div(pool)
        _emit_div(lines, bm_pool)
        ranked = rank_by_iv(pool, run_dir)
        for k in TOPK:
            sub = pool if k >= len(pool) else ranked[:k]
            _emit_div(lines, bm_pool if sub is pool else _div(sub),
                      prefix="topk_", k=k, n=len(sub))
    audit_path = Path(run_dir) / "prune_audit.jsonl"
    if audit_path.is_file():
        import json as _json
        orig, final, triggered = [], [], 0
        for ln in audit_path.read_text(encoding="utf-8").splitlines():
            try:
                e = _json.loads(ln)
            except Exception:
                continue
            if e.get("orig_tokens") is not None:
                orig.append(float(e["orig_tokens"]))
            if e.get("final_tokens") is not None:
                final.append(float(e["final_tokens"]))
                if e.get("orig_tokens") and float(e["orig_tokens"]) > float(e["final_tokens"]):
                    triggered += 1
        if orig:
            lines.append(emit("subgraph_tokens", dim="orig_mean",
                              value=round(sum(orig) / len(orig), 1), n=len(orig)))
            lines.append(emit("subgraph_tokens", dim="final_mean",
                              value=round(sum(final) / len(final), 1), n=len(final)))
            lines.append(emit("prune_triggered_ratio",
                              value=round(triggered / len(final), 4),
                              n=len(final)))
    usd = _usd_cost(usage.get("prompt_tokens"), usage.get("completion_tokens"))
    lines.append(emit("cost", llm_calls=usage.get("calls"),
                      prompt_tokens=usage.get("prompt_tokens"),
                      completion_tokens=usage.get("completion_tokens"),
                      usd=usd,
                      graph_queries=usage.get("graph_queries")))
    n_all = len(pool_all)

    def _per(v):
        return round(float(v) / n_all, 2) if (v is not None and n_all) else None

    lines.append(emit("cost_per_valid", llm_calls=_per(usage.get("calls")),
                      prompt_tokens=_per(usage.get("prompt_tokens")),
                      completion_tokens=_per(usage.get("completion_tokens")),
                      usd=(round(usd / n_all, 4) if (usd is not None and n_all) else None),
                      graph_queries=_per(usage.get("graph_queries")),
                      n_valid=n_all))
    return lines


def _usd_cost(prompt_tokens, completion_tokens) -> float | None:
    try:
        p_in = float(os.environ["LLM_PRICE_INPUT_PER_M"])
        p_out = float(os.environ["LLM_PRICE_OUTPUT_PER_M"])
    except (KeyError, ValueError):
        return None
    if prompt_tokens is None or completion_tokens is None:
        return None
    return round((float(prompt_tokens) * p_in + float(completion_tokens) * p_out) / 1e6, 4)


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m experiments.metrics",
        description="Metrics 1 and 2 + five rates + cost → runs/<run_id>/metrics.txt")
    ap.add_argument("run_dirs", nargs="+", help="Run directories (one or more)")
    ap.add_argument("--pool-limit", type=int, default=0,
                    help="Use only the first N features.json entries in file order; 0=full pool. "
                         "When >0, process metrics also stop at the episode containing entry N")
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel processes for pairwise GED (0=automatic min(8,CPU); "
                         "override the automatic value with FRAUDFEAT_GED_WORKERS)")
    ap.add_argument("--reuse-div", action="store_true",
                    help="Reuse structural diversity lines from existing metrics.txt in the same directory when counts match; do not recompute GED")
    args = ap.parse_args(argv)
    global _WORKERS, _REUSE_DIV
    _WORKERS = args.workers or None
    _REUSE_DIV = bool(args.reuse_div)
    for d in args.run_dirs:
        lines = compute_metrics(d, pool_limit=args.pool_limit)
        out = write_txt(d, "metrics.txt", lines)
        print(f"[metrics] {d} → {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main(sys.argv[1:])
