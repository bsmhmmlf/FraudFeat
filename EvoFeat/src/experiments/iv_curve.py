from __future__ import annotations

import sys
from pathlib import Path

from ._common import effective_iv, emit, read_history, read_pool, write_txt
from .structural_diversity import batch_metrics


def compute_curves(run_dir: str | Path) -> list[str]:
    run_dir = Path(run_dir)
    lines: list[str] = []
    for history in read_history(run_dir):
        stats = history.get("episode_stats") or {}
        if not stats:
            continue
        episode = history.get("episode")
        lines.append(emit("evaluator_tier_acc", episode=episode, span="inc",
                          value=stats.get("tier_acc"), n=stats.get("n_grounded")))
        lines.append(emit("evaluator_abs_div", episode=episode, span="inc",
                          value=stats.get("avg_abs_div"), n=stats.get("n_grounded")))
    by_episode: dict[int, list[dict]] = {}
    for feature in read_pool(run_dir, limit=0):
        episode = feature.get("episode")
        if episode is not None:
            by_episode.setdefault(int(episode), []).append(feature)
    cumulative: list[dict] = []
    for episode in sorted(by_episode):
        cumulative.extend(by_episode[episode])
        metrics = batch_metrics(cumulative, curve_only=True)
        lines.append(emit("explorer_pool_families", episode=episode, span="cum",
                          value=metrics["distinct_structures"], n=metrics["n"]))
        lines.append(emit("explorer_pool_struct_sim", episode=episode, span="cum",
                          value=metrics["avg_pairwise_sim"], n=metrics["n"],
                          n_pairs=metrics["n_pairs"]))
        lines.append(emit("explorer_pool_near_dup", episode=episode, span="cum",
                          value=metrics["near_dup_ratio"], n=metrics["n"]))
        ivs = [effective_iv(feature.get("eval_iv"), run_dir) for feature in cumulative]
        ivs = sorted((value for value in ivs if value is not None), reverse=True)[:100]
        lines.append(emit("explorer_pool_top100_iv", episode=episode, span="cum",
                          value=(sum(ivs) / len(ivs) if ivs else None),
                          n=len(ivs), pool_size=len(cumulative)))
    return lines


def main(argv: list[str]) -> None:
    for run_dir in argv:
        lines = compute_curves(run_dir)
        out = write_txt(run_dir, "iv_curve.txt", lines)
        print(f"[iv_curve] {run_dir} -> {out} ({len(lines)} rows)")


if __name__ == "__main__":
    main(sys.argv[1:])
