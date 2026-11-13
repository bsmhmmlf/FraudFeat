from __future__ import annotations

from . import family


ANCHOR_PER_FAMILY = 6
ANCHOR_MAX_TOTAL = 40


_EPS = 0.02


_DIR_TXT = {"out": "outgoing", "in": "incoming"}


def family_key(f) -> str:
    rels, terminal, evt = family.structure_signature(f)[:3]
    return _compose_key(rels, terminal, evt)


def family_key_loose(f) -> str:
    rels, terminal, _evt = family.structure_signature(f)[:3]
    return _compose_key(rels, terminal, "")


def _compose_key(rels, terminal: str, evt: str) -> str:
    rels_txt = "→".join(rels) if rels else "?"
    key = f"{rels_txt} | terminal={terminal or '?'}"
    if evt:
        key += f" | value_type={evt}"
    return key


def _delta(pred, true):
    try:
        if pred is None or true is None:
            return None
        return float(pred) - float(true)
    except (TypeError, ValueError):
        return None


def _delta_txt(d) -> str:
    if d is None:
        return "—"
    sign = "+" if d > 0 else ""
    if abs(d) < _EPS:
        tag = "well-calibrated"
    elif d > 0:
        tag = "overestimated"
    else:
        tag = "underestimated"
    return f"{sign}{d:.3f} ({tag})"


def _chain_txt(chain) -> str:
    segs: list[str] = []
    for i, s in enumerate(chain or []):
        if not isinstance(s, dict):
            continue
        rel = str(s.get("rel") or s.get("rel_name") or s.get("join_field") or "?").strip()
        d = _DIR_TXT.get(str(s.get("direction") or "").strip(), "")
        seg = (f"{s.get('from', '?')} -[{rel}"
               + (f" | {d}" if d else "") + f"]-> {s.get('to', '?')}")
        note = str(s.get("note") or "").strip()
        if note:
            seg += f" ({note})"
        segs.append(f"segment {i + 1}: {seg}")
    return "; ".join(segs) or "(no evidence chain)"


def _final_txt(plan) -> str:
    if not isinstance(plan, dict):
        return "—"
    final = plan.get("final") if isinstance(plan.get("final"), dict) else {}
    if not final:
        return "—"
    txt = str(final.get("type") or "?").strip()
    op = str(final.get("op") or "").strip()
    if op:
        txt += f"/{op}"
    return txt


def _sample_note(cand) -> str:
    parts: list[str] = []
    stratum = getattr(cand, "_stratum", None)
    if stratum:
        parts.append(f"stratum={stratum}")
    return "; ".join(parts)


def _paradigm_txt(cand) -> str:
    try:
        from . import paradigm
        slug = paradigm.paradigm_of(cand)
        if not slug or slug == paradigm.UNCLASSIFIED:
            return ""
        return paradigm.display_of(slug)
    except Exception:
        return ""


def _entry_block(idx: int, item: dict, episode: int) -> str:
    cand = item.get("candidate")
    name = str(item.get("doc_name") or getattr(cand, "feature_name", "") or "?")
    if cand is None:
        return f"{idx}. {name} (material missing: absorb returned no candidate)"
    cn = str(getattr(cand, "feature_name_cn", "") or "")
    chain = getattr(cand, "evidence_chain", None) or []
    hops = len(chain)
    pred = getattr(cand, "predicted_iv", None)
    true = getattr(cand, "true_iv", None)
    d = _delta(pred, true)
    lines = [f"{idx}. {name}" + (f" ({cn})" if cn else "")
             + f" | episode {episode}" + (" | new this episode" if item.get("is_new") else " | updated this episode")]
    lines.append(f"   Structure: hops={hops}; {_chain_txt(chain)}")
    lines.append(f"   Caliber: expected_value_type={getattr(cand, 'expected_value_type', '') or 'undeclared'}"
                 f"; actual_final_reduction={_final_txt(getattr(cand, 'gremlin_plan', None))}")
    lines.append(f"   Prediction: iv={_num(pred)} tier="
                 f"{getattr(getattr(cand, 'predicted_tier', None), 'value', '?')}"
                 f" conf={_num(getattr(cand, 'predicted_conf', None))}")
    reason = str(getattr(cand, "predicted_reason", "") or "").strip()
    lines.append(f"   Evaluator rationale: {reason or '(none)'}")
    lines.append(f"   Measured: iv={_num(true)}"
                 f" | delta={_delta_txt(d)}")
    sample = _sample_note(cand)
    if sample:
        lines.append(f"   Sampling: {sample}")
    consistency = str(getattr(cand, "consistency_report", "") or "")
    lines.append("   Consistency: " + ("not validated; diagnosis incomplete, so measured IV has reduced credibility"
                                 if "not validated" in consistency else "validated"))
    para = _paradigm_txt(cand)
    if para:
        lines.append(f"   Paradigm:{para}")
    return "\n".join(lines)


def _num(v) -> str:
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "—"


def _anchor_rows(group_key: str, rows: list, limit: int) -> str:
    head = (f"Family [{group_key}]: {len(rows)} historical rows; showing {min(limit, len(rows))} "
            "(raw rows without aggregation)")
    body = []
    for m in rows[:limit]:
        name = str(getattr(m, "doc_name", "") or "?")
        true = getattr(m, "true_iv", None)
        pred = getattr(m, "predicted_iv", None)
        conf = getattr(m, "predicted_conf", None)
        txt = (f"  - {name} (measured IV {_num(true)} | prediction {_num(pred)}"
               f" | delta {_delta_txt(_delta(pred, true))}")
        if conf is not None:
            txt += f" | conf_then {_num(conf)}"
        body.append(txt + ")")
    return head + "\n" + "\n".join(body)


def build_material(written: list, all_memory: list, episode: int) -> str:
    items = [x for x in (written or []) if isinstance(x, dict)]
    if not items:
        return ""
    written_names = {str(x.get("doc_name") or "") for x in items}

    def _sort_key(x):
        d = _delta(getattr(x.get("candidate"), "predicted_iv", None),
                   getattr(x.get("candidate"), "true_iv", None))
        return (-abs(d) if d is not None else 0.0)

    items = sorted(items, key=_sort_key)
    blocks = [_entry_block(i + 1, x, episode) for i, x in enumerate(items)]
    buckets: dict[str, list] = {}
    for m in (all_memory or []):
        name = str(getattr(m, "doc_name", "") or "")
        if name and name in written_names:
            continue
        key = family_key_loose(m)
        buckets.setdefault(key, []).append(m)
    anchor_blocks: list[str] = []
    used = 0
    seen_keys: set[str] = set()
    for x in items:
        cand = x.get("candidate")
        if cand is None:
            continue
        key = family_key_loose(cand)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rows = buckets.get(key) or []
        if not rows or used >= ANCHOR_MAX_TOTAL:
            continue
        limit = min(ANCHOR_PER_FAMILY, ANCHOR_MAX_TOTAL - used)
        anchor_blocks.append(_anchor_rows(key, rows, limit))
        used += min(limit, len(rows))
    out = [
        f"[ALL {len(items)} IV-RAG ENTRIES WRITTEN IN THE CURRENT EPISODE] "
        f"The system only assembles raw material without aggregation. Compare entries and derive the lessons yourself.",
        "\n\n".join(blocks),
    ]
    if anchor_blocks:
        out.append("[HISTORICAL ANCHORS: same-family IV-RAG rows excluding current entries; raw rows only] "
                   "Use them to distinguish isolated cases from repeated bias.")
        out.append("\n\n".join(anchor_blocks))
    else:
        out.append("[HISTORICAL ANCHORS] No same-family historical rows exist for the current entries. "
                   "Treat this as a new family and avoid creating a rule from insufficient evidence.")
    return "\n\n".join(out)
