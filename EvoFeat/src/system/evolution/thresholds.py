from __future__ import annotations

from dataclasses import dataclass, field

from ..utils import get_logger


logger = get_logger("thresholds")


def _safe_iv(x: dict) -> float:
    try:
        return float(x.get("iv", x.get("miv", 0.0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


DEFAULT_FIXED = {
    "positive_iv": 0.20,
    "redundant": 0.80,
    "min_avg_iv": 0.20,
    "skill_hit": 0.20,
}
DEFAULT_ADAPTIVE = {
    "pos_quantile": 0.70,
    "neg_quantile": 0.30,
    "pos_cap": 0.30,
    "neg_cap": 0.30,
    "min_pos": 1,
    "min_neg": 1,
}


@dataclass
class ThresholdPolicy:
    mode: str = "fixed"
    fixed: dict = field(default_factory=lambda: dict(DEFAULT_FIXED))
    adaptive: dict = field(default_factory=lambda: dict(DEFAULT_ADAPTIVE))

    @classmethod
    def from_config(cls, cfg: dict | None) -> "ThresholdPolicy":
        cfg = cfg or {}
        mode = str(cfg.get("mode", "fixed")).lower()
        if mode not in ("fixed", "adaptive"):
            logger.warning(f"Unknown threshold mode {mode}; falling back to fixed")
            mode = "fixed"
        fixed = dict(DEFAULT_FIXED)
        fixed.update(cfg.get("fixed", {}) or {})
        adaptive = dict(DEFAULT_ADAPTIVE)
        adaptive.update(cfg.get("adaptive", {}) or {})
        return cls(mode=mode, fixed=fixed, adaptive=adaptive)

    def threshold(self, key: str) -> float:
        if self.mode == "adaptive":
            return float(self.adaptive.get(key, DEFAULT_FIXED.get(key, 0.15)))
        return float(self.fixed.get(key, DEFAULT_FIXED.get(key, 0.15)))

    def redundant_threshold(self) -> float:
        return self.threshold("redundant")

    def min_avg_iv(self) -> float:
        return self.threshold("min_avg_iv")

    def skill_hit_threshold(self) -> float:
        return self.threshold("skill_hit")

    def label_samples(self, items: list[dict]) -> list[dict]:
        if not items:
            return []
        items = [dict(x) for x in items]
        ivs = [_safe_iv(x) for x in items]
        has_iv = any(x.get("iv") is not None or x.get("miv") is not None for x in items)
        if not has_iv:
            return items
        if self.mode == "adaptive":
            pos_q = float(self.adaptive.get("pos_quantile", 0.70))
            neg_q = float(self.adaptive.get("neg_quantile", 0.30))
            sorted_iv = sorted(ivs)
            n = len(sorted_iv)
            pos_thr = sorted_iv[min(n - 1, max(0, int(n * pos_q)))]
            neg_thr = sorted_iv[max(0, min(n - 1, int(n * neg_q) - 1))]
            for x in items:
                mv = _safe_iv(x)
                if mv >= pos_thr:
                    x["positive"] = True
                    x["negative"] = False
                elif mv <= neg_thr:
                    x["positive"] = False
                    x["negative"] = True
                else:
                    x["positive"] = False
                    x["negative"] = False
        else:
            pos_thr = float(self.fixed.get("positive_iv", 0.20))
            for x in items:
                is_pos = _safe_iv(x) >= pos_thr
                x["positive"], x["negative"] = is_pos, not is_pos
        n_pos = sum(1 for x in items if x.get("positive"))
        n_neg = sum(1 for x in items if x.get("negative"))
        logger.info(f"[Thresholds] Sample labels: positive {n_pos} / negative {n_neg} / total {len(items)}"
                    f" (mode={self.mode}; paper convention: IV>=0.2 positive, otherwise negative)")
        return items

    def to_dict(self) -> dict:
        return {"mode": self.mode, "fixed": self.fixed, "adaptive": self.adaptive}
