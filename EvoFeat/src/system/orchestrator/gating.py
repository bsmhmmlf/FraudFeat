from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..utils import get_logger


logger = get_logger("gating")


@dataclass
class GateConfig:
    queue_threshold: int = 8
    budget_per_episode: int = 10
    iv_reference: float = 0.4
    iv_weight: float = 1.0
    blind_weight: float = 2.0
    strat_high: float = 0.8
    strat_low: float = 0.2


class Gate:
    def __init__(self, cfg: GateConfig | None = None):
        self.cfg = cfg or GateConfig()

    def should_ground(self, queue_size: int, force_budget: bool) -> tuple[bool, str]:
        if queue_size >= self.cfg.queue_threshold:
            return True, f"Queue threshold ({queue_size}>={self.cfg.queue_threshold})"
        if force_budget:
            return True, "Per-episode measurement budget"
        return False, ""

    def _score(self, candidate: Any) -> float | None:
        iv = getattr(candidate, "predicted_iv", None)
        confidence = getattr(candidate, "predicted_conf", None)
        if iv is None or confidence is None:
            return None
        iv = max(0.0, float(iv))
        confidence = min(1.0, max(0.0, float(confidence)))
        denominator = iv + self.cfg.iv_reference
        numerator = abs(iv - self.cfg.iv_reference)
        if denominator == 0.0:
            extremity = 0.0 if numerator == 0.0 else float("inf")
        else:
            extremity = numerator / denominator
        return (self.cfg.iv_weight * extremity
                + self.cfg.blind_weight * (1.0 - confidence))

    def _ordered(self, candidates: list, reverse: bool = True) -> list:
        indexed = list(enumerate(candidates))
        valid = [(self._score(c), i, c) for i, c in indexed]
        scored = [(s, i, c) for s, i, c in valid if s is not None]
        scored.sort(key=lambda row: ((-row[0]) if reverse else row[0], row[1]))
        return [c for _, _, c in scored]

    def select_top_k(self, candidates: list, k: int) -> list:
        if not candidates or k <= 0:
            return []
        picked = self._ordered(candidates, reverse=True)[:k]
        if len(picked) < k:
            selected = {id(c) for c in picked}
            picked.extend(c for c in candidates if id(c) not in selected)
            picked = picked[:k]
        return picked

    def select_stratified(self, candidates: list, k: int) -> list:
        if not candidates or k <= 0:
            return []
        k = min(k, len(candidates))
        n_high = min(k, max(1, int(round(k * self.cfg.strat_high))))
        n_low = min(k - n_high, int(round(k * self.cfg.strat_low)))
        high = self._ordered(candidates, reverse=True)
        low = self._ordered(candidates, reverse=False)
        picked = []
        seen = set()

        def take(sequence, count, stratum):
            for candidate in sequence:
                if count <= 0:
                    break
                if id(candidate) in seen:
                    continue
                seen.add(id(candidate))
                candidate._stratum = stratum
                picked.append(candidate)
                count -= 1

        take(high, n_high, "high")
        take(low, n_low, "low")
        if len(picked) < k:
            take(high, k - len(picked), "topup")
        if len(picked) < k:
            take(candidates, k - len(picked), "topup")
        logger.info(
            f"Selected {len(picked)}/{k} candidates using the paper score "
            f"(high={n_high}, low={n_low}, IV0={self.cfg.iv_reference}, "
            f"alpha={self.cfg.iv_weight}, beta={self.cfg.blind_weight})"
        )
        return picked

    def select_random(self, candidates: list, k: int,
                      seed: int | None = None) -> list:
        if not candidates or k <= 0:
            return []
        import random
        rng = random.Random(seed)
        picked = list(candidates)
        rng.shuffle(picked)
        picked = picked[:k]
        for candidate in picked:
            candidate._stratum = "random"
        return picked
