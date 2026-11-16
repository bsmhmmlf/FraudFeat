from __future__ import annotations

import sys
from pathlib import Path

from ._common import (analysis_cutoff, emit, read_candidates_reconciled, read_pool,
                      within_cutoff, write_txt)


TOPK_HIT = 10


def _tier(iv: float) -> str:
    return "high" if iv >= 0.5 else ("mid" if iv >= 0.2 else "low")


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else round((xs[n // 2 - 1] + xs[n // 2]) / 2, 4)


def compute_calibration(run_dir: str | Path, pool_limit: int = 0) -> list[str]:
    lines: list[str] = []
    cutoff = analysis_cutoff(read_pool(run_dir), pool_limit)
    cands = [c for c in read_candidates_reconciled(run_dir) if within_cutoff(c, cutoff)]
    if cutoff is not None:
        lines.append(emit("calibration_cutoff", episode=cutoff, pool_limit=pool_limit))
    graded = [c for c in cands
              if c.get("true_iv") is not None and c.get("predicted_iv") is not None]
    by_stratum_all: dict[str, int] = {}
    by_stratum_graded: dict[str, list[dict]] = {}
    for c in cands:
        s = c.get("stratum") or "unselected"
        by_stratum_all[s] = by_stratum_all.get(s, 0) + 1
        if c.get("true_iv") is not None:
            by_stratum_graded.setdefault(s, []).append(c)
    for s, n_all in sorted(by_stratum_all.items()):
        n_g = len(by_stratum_graded.get(s, []))
        lines.append(emit("stratum_sampling_rate", stratum=s,
                          value=(round(n_g / n_all, 4) if n_all else None),
                          n_graded=n_g, n_total=n_all))
    for s, rows in sorted(by_stratum_graded.items()):
        rows = [r for r in rows if r.get("predicted_iv") is not None]
        if not rows:
            continue
        acc = sum(1 for r in rows
                  if _tier(float(r["predicted_iv"])) == _tier(float(r["true_iv"])))
        signed = [float(r["predicted_iv"]) - float(r["true_iv"]) for r in rows]
        absd = [abs(x) for x in signed]
        lines.append(emit("calibration", stratum=s, n=len(rows),
                          tier_acc=round(acc / len(rows), 4),
                          signed_div=round(sum(signed) / len(signed), 4),
                          abs_div_mean=round(sum(absd) / len(absd), 4),
                          abs_div_median=_median(absd)))
    if graded:
        signed = [float(c["predicted_iv"]) - float(c["true_iv"]) for c in graded]
        absd = [abs(x) for x in signed]
        lines.append(emit("calibration", stratum="all", n=len(graded),
                          signed_div=round(sum(signed) / len(signed), 4),
                          abs_div_mean=round(sum(absd) / len(absd), 4),
                          abs_div_median=_median(absd)))
    if len(graded) >= TOPK_HIT:
        by_pred = sorted(graded, key=lambda c: -float(c["predicted_iv"]))[:TOPK_HIT]
        by_true = sorted(graded, key=lambda c: -float(c["true_iv"]))[:TOPK_HIT]
        hit = len({c["feature_name"] for c in by_pred}
                  & {c["feature_name"] for c in by_true})
        lines.append(emit("topk_hit_rate", k=TOPK_HIT,
                          value=round(hit / TOPK_HIT, 4)))
    return lines


def main(argv: list[str]) -> None:
    pool_limit = 0
    dirs: list[str] = []
    it = iter(argv)
    for a in it:
        if a == "--pool-limit":
            pool_limit = int(next(it))
        elif a.startswith("--pool-limit="):
            pool_limit = int(a.split("=", 1)[1])
        else:
            dirs.append(a)
    for d in dirs:
        lines = compute_calibration(d, pool_limit)
        out = write_txt(d, "calibration.txt", lines)
        print(f"[calibration] {d} → {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main(sys.argv[1:])
