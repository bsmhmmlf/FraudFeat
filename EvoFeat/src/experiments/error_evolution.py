from __future__ import annotations

import random
import sys
from pathlib import Path

from ._common import emit, read_candidates_reconciled, write_txt


CLASSES = ("relation", "depth", "aggregate", "exec", "consistency",
           "query_form", "unknown")
_BOOT_N = 1000
_BOOT_SEED = 20260817


def _ols_slope(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def _boot_slope_ci(xs: list[float], ys: list[float]) -> tuple[float | None, float | None]:
    n = len(xs)
    if n < 3:
        return None, None
    rng = random.Random(_BOOT_SEED)
    slopes = []
    for _ in range(_BOOT_N):
        idx = [rng.randrange(n) for _ in range(n)]
        s = _ols_slope([xs[i] for i in idx], [ys[i] for i in idx])
        if s is not None:
            slopes.append(s)
    if not slopes:
        return None, None
    slopes.sort()
    lo = slopes[int(0.025 * len(slopes))]
    hi = slopes[min(len(slopes) - 1, int(0.975 * len(slopes)))]
    return round(lo, 5), round(hi, 5)


def compute_error_evolution(run_dir: str | Path) -> list[str]:
    lines: list[str] = []
    cands = read_candidates_reconciled(run_dir)
    by_w: dict[int, list[dict]] = {}
    for c in cands:
        w = c.get("episode")
        if w is not None:
            by_w.setdefault(int(w), []).append(c)
    episodes = sorted(by_w)
    seen_sigs: set[str] = set()
    rate_x: list[float] = []
    rate_y: list[float] = []
    for w in episodes:
        rows = by_w[w]
        n = len(rows)
        class_cnt = {k: 0 for k in CLASSES}
        n_err = 0
        for c in rows:
            sig = (c.get("error_signature") or "").strip()
            if sig:
                n_err += 1
                for k in sig.split("+"):
                    if k in class_cnt:
                        class_cnt[k] += 1
        for k in CLASSES:
            lines.append(emit("error_class", episode=w, cls=k,
                              count=class_cnt[k],
                              rate=(round(class_cnt[k] / n, 4) if n else None)))
        lines.append(emit("error_rate", episode=w, count=n_err, denom=n,
                          value=(round(n_err / n, 4) if n else None)))

        cur_sigs = {(c.get("error_signature") or "") + "@" + (c.get("sig") or "")
                    for c in rows if c.get("error_signature")}
        recur = sum(1 for s in cur_sigs if s in seen_sigs)
        lines.append(emit("error_recurrence", episode=w,
                          value=(round(recur / len(cur_sigs), 4) if cur_sigs else None),
                          recurring=recur, seen=len(seen_sigs), current=len(cur_sigs)))
        seen_sigs |= cur_sigs
        repaired = [c for c in rows if (c.get("repairs") or 0) > 0]
        ok1 = sum(1 for c in repaired
                  if c.get("repairs") == 1 and c.get("outcome") == "pooled")
        lines.append(emit("first_repair_success", episode=w,
                          value=(round(ok1 / len(repaired), 4) if repaired else None),
                          n_repaired=len(repaired)))
        if n:
            rate_x.append(float(w))
            rate_y.append(n_err / n)
    slope = _ols_slope(rate_x, rate_y)
    lo, hi = _boot_slope_ci(rate_x, rate_y)
    lines.append(emit("error_rate_trend", slope=(round(slope, 5)
                                                 if slope is not None else None),
                      ci95_lo=lo, ci95_hi=hi, n_episodes=len(rate_x)))
    return lines


def main(argv: list[str]) -> None:
    for d in argv:
        lines = compute_error_evolution(d)
        out = write_txt(d, "error_evolution.txt", lines)
        print(f"[error_evolution] {d} → {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main(sys.argv[1:])
