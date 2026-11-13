from __future__ import annotations

import json
from pathlib import Path

from ..paths import MEMORY_DIR
from ..utils import get_logger
from . import paradigm
from .family import final_agg_method, get_field


logger = get_logger("diversity")
DEFAULT_STATE_PATH = MEMORY_DIR / "diversity_state.json"


CALIBER_TYPES = ("count", "flag", "amount", "ratio")
CALIBER_TYPE_LABELS = {"count": "count", "flag": "flag",
                       "amount": "amount", "ratio": "ratio"}
CALIBER_AGG_LABELS = {"sum": "sum", "mean": "mean", "min": "min", "max": "max"}
CALIBER_SCARCE_MAX = 3


def count_calibers(pool) -> tuple[dict[str, int], dict[str, int]]:
    types: dict[str, int] = {}
    aggs: dict[str, int] = {}
    for f in pool or []:
        vt = str(get_field(f, "value_type", "") or "").strip().lower()
        if not vt:
            vt = str(get_field(f, "expected_value_type", "") or "").strip().lower()
        if vt:
            types[vt] = types.get(vt, 0) + 1
        agg = final_agg_method(f)
        if agg in CALIBER_AGG_LABELS:
            aggs[agg] = aggs.get(agg, 0) + 1
    return types, aggs


def _caliber_block(types: dict[str, int], aggs: dict[str, int]) -> str:
    if not types and not aggs:
        return ""
    lines = ["[OUTPUT TYPE AND AGGREGATION COVERAGE — SOFT STRUCTURAL GUIDANCE]"]
    for k in CALIBER_TYPES:
        n = types.get(k, 0)
        if n == 0:
            lines.append(f"  - {CALIBER_TYPE_LABELS[k]}: 0 (uncovered; prioritize)")
        elif n <= CALIBER_SCARCE_MAX:
            lines.append(f"  - {CALIBER_TYPE_LABELS[k]}: {n} (scarce; prioritize)")
        else:
            lines.append(f"  - {CALIBER_TYPE_LABELS[k]}: {n}")
    covered = [f"{CALIBER_AGG_LABELS[k]}={aggs[k]}"
               for k in CALIBER_AGG_LABELS if aggs.get(k)]
    missing = [CALIBER_AGG_LABELS[k] for k in CALIBER_AGG_LABELS if not aggs.get(k)]
    lines.append(f"  - Aggregations: {', '.join(covered) or 'none'}"
                 + (f"; uncovered={', '.join(missing)}" if missing else ""))
    lines.append("  A different output type or aggregation is a new caliber; structural coverage remains the primary guidance.")
    return "\n".join(lines)


class ParadigmDiversityTracker:
    def __init__(self, path: str | Path | None = None, persist: bool = True):
        self.path = Path(path) if path else DEFAULT_STATE_PATH
        self.persist = persist
        self._saturated: set[str] = set()
        self._counts: dict[str, int] = {}
        self._calibers: dict[str, int] = {}
        self._aggs: dict[str, int] = {}
        self.updated_episode: int = 0
        if persist:
            self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            raw = {str(x) for x in data.get("saturated") or []}
            self.updated_episode = int(
                data.get("updated_episode", data.get("updated_window", 0)) or 0)
        except (OSError, ValueError) as e:
            logger.warning(f"[Diversity] Failed to load saturation list ({e}); starting from scratch")
            return

        valid = paradigm.valid_slugs()
        if valid:
            stale = {s for s in raw if s not in valid}
            if stale:
                logger.warning(f"[Diversity] Discarded {len(stale)} state entries absent from the current paradigm set "
                               f"(mostly legacy structural groups; no migration): {sorted(stale)}")
            raw -= stale
        self._saturated = raw
        if self._saturated:
            logger.info(f"[Diversity] Loaded saturated paradigms: {sorted(self._saturated)}"
                        f" (through episode {self.updated_episode})")

    def save(self) -> None:
        if not self.persist:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(
                {"saturated": sorted(self._saturated),
                 "updated_episode": self.updated_episode},
                ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError as e:
            logger.warning(f"[Diversity] Failed to write saturation list ({e}): {self.path}")

    def reset(self) -> None:
        self._saturated = set()
        self._counts = {}
        self._calibers = {}
        self._aggs = {}
        self.updated_episode = 0
        if self.persist and self.path.exists():
            try:
                self.path.unlink()
            except OSError as e:
                logger.warning(f"[Diversity] Failed to delete saturation list ({e})")

    def update_from_pool(self, pool, *, episode: int = 0) -> set[str]:
        counts = paradigm.count_by_paradigm(pool)
        counts.pop(paradigm.UNCLASSIFIED, None)
        self._counts = counts
        self._calibers, self._aggs = count_calibers(pool)
        newly = {slug for slug, n in counts.items() if paradigm.is_saturated(n)}
        added = newly - self._saturated
        if added:
            self._saturated |= newly
            logger.info(f"[Diversity] Newly saturated paradigms: "
                        f"{[paradigm.title_of(s) for s in sorted(added)]}"
                        f" (pool feature count >={paradigm.SATURATED_MIN}; "
                        f"next-episode navigation will prohibit renamed variants of the same paradigm)")
        if counts:
            dist = ", ".join(f"{paradigm.title_of(s)} {n}"
                            for s, n in sorted(counts.items(), key=lambda kv: -kv[1]))
            logger.info(f"[Diversity] Pool paradigm coverage: {dist}")
        if self._calibers or self._aggs:
            vt = ", ".join(f"{k} {n}" for k, n in sorted(self._calibers.items(),
                                                        key=lambda kv: -kv[1])) or "none"
            ag = ", ".join(f"{k} {n}" for k, n in sorted(self._aggs.items(),
                                                         key=lambda kv: -kv[1])) or "none"
            logger.info(f"[Diversity] Pool caliber coverage: {vt} (aggregations: {ag})")
        if episode:
            self.updated_episode = max(self.updated_episode, episode)
        if added or episode:
            self.save()
        return set(self._saturated)

    def saturated(self) -> set[str]:
        return set(self._saturated)

    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def guidance_block(self) -> str:
        specs = paradigm.active()
        if not specs:
            return ""
        rows: list[str] = []
        hard: list[str] = []
        low: list[str] = []
        for spec in specs:
            n = int(self._counts.get(spec.slug, 0) or 0)
            key, label, guidance = paradigm.coverage_tier(n)
            if key == "hard" or spec.slug in self._saturated:
                key, label, guidance = paradigm.SOFT_TIERS[-1][1:]
                hard.append(spec.title or spec.slug)
            elif key in ("none", "light"):
                low.append(spec.title or spec.slug)
            rows.append(f"  - {spec.display} | pool_count={n} ({label}) -> {guidance}")
        lines = ["[STRUCTURAL PARADIGM COVERAGE AND GENERATION GUIDANCE "
                 "based on actual feature-pool coverage]"]
        lines.extend(rows)
        if hard:
            lines.append(f"  Saturated paradigms ({', '.join(hard)}): do not produce renamed variants of the same paradigm. "
                         f"A variant counts as new only if it changes the relation sequence, terminal entity, value type, or aggregation.")
        if low:
            lines.append(f"  Priority directions: {', '.join(low)}"
                         f" (least covered; prioritize these under equal conditions).")
        lines.append("  Structures not assigned to an existing paradigm are unrestricted; discovering structures beyond current coverage is especially valuable. "
                     "")
        calibers = _caliber_block(self._calibers, self._aggs)
        if calibers:
            lines.append(calibers)
        return "\n".join(lines) + "\n"


SaturatedFamilyTracker = ParadigmDiversityTracker
