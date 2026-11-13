from __future__ import annotations

from ..utils import get_logger
from . import paradigm
from .diversity import _caliber_block, count_calibers


logger = get_logger("explore_guidance")
MAX_ROWS = 8
HIGH_IV = 0.5


def _verdict(tier_key: str, has_high_iv: bool) -> tuple[str, str]:
    if tier_key == "hard":
        if has_high_iv:
            return ("LIMITED", "This paradigm is saturated but has high measured IV. Only genuinely new "
                               "variants (a different mediator, terminal entity, value type, or aggregation) are allowed.")
        return ("DEPRIORITIZE", "This paradigm is saturated without high-IV evidence. Do not produce further "
                                "variants; move to another structure.")
    if tier_key in ("none", "light"):
        return "PRIORITIZE", "This structure has the least coverage and should receive exploration budget first."
    if tier_key == "strong":
        return "DISCOURAGE", "This structure is nearly saturated; avoid repetitive variants unless the angle is clearly new."
    return "NEUTRAL", "This structure is already represented; continue only with a distinct observation angle."


def _measured_iv(f) -> float | None:
    v = getattr(f, "true_iv", None)
    if v is None and isinstance(f, dict):
        v = f.get("true_iv")
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def build_guidance_block(*, selected: list | None = None, pool: list | None = None,
                         tracker=None, enable_coverage: bool = True) -> str:
    if not enable_coverage:
        return ""
    pool = list(pool or [])
    counts: dict[str, int] = {}
    high: dict[str, bool] = {}
    for f in pool:
        if not (getattr(f, "evidence_chain", None) or []):
            continue
        slug = paradigm.paradigm_of(f)
        counts[slug] = counts.get(slug, 0) + 1
        iv = _measured_iv(f)
        if iv is not None and iv >= HIGH_IV:
            high[slug] = True
    rows = []
    for spec in paradigm.active():
        count = int(counts.get(spec.slug, 0) or 0)
        tier_key, label, _tone = paradigm.coverage_tier(count)
        verdict, sentence = _verdict(tier_key, bool(high.get(spec.slug)))
        rank = {"none": 0, "light": 0, "medium": 1, "strong": 2, "hard": 3}.get(tier_key, 1)
        rows.append((rank, f"  - {spec.display} | pool count {count} ({label}) -> [{verdict}] {sentence}"))
    unclassified = int(counts.get(paradigm.UNCLASSIFIED, 0) or 0)
    rows.append((0, f"  - Unclassified structures | pool count {unclassified} | open: structures outside "
                    f"all recorded paradigms remain valuable exploration targets"))
    lines = [
        "[COVERAGE GUIDANCE]",
        f"Overview: {len(pool)} pooled features | {sum(1 for s in counts if s != paradigm.UNCLASSIFIED)} "
        f"covered paradigms | {unclassified} unclassified | a paradigm with >= {paradigm.SATURATED_MIN} "
        f"features is saturated",
    ]
    rows.sort(key=lambda row: (row[0], row[1]))
    lines.extend(text for _, text in rows[:MAX_ROWS])
    if len(rows) > MAX_ROWS:
        lines.append(f"  ... {len(rows) - MAX_ROWS} additional classes omitted")
    types, aggs = count_calibers(pool)
    calibers = _caliber_block(types, aggs)
    if calibers:
        lines.append(calibers)
    block = "\n".join(lines) + "\n"
    logger.info(f"[guidance] rows={len(rows)} chars={len(block)}")
    return block
