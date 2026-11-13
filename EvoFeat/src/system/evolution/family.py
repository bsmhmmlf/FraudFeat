from __future__ import annotations

import json
import re


def get_field(f, key, default=None):
    if isinstance(f, dict):
        v = f.get(key)
    else:
        v = getattr(f, key, None)
    return default if v is None else v


def jf_short(jf: str) -> str:
    jf = (jf or "").strip()
    if not jf:
        return ""
    left, _, right = jf.partition("|")
    segs = [s.strip() for s in left.split("&") if s.strip()]
    short = "&".join(segs[:2])
    tail = right.strip()
    if tail in ("payer", "payee"):
        return f"{short}|{tail}"
    return short


def rel_short(chain) -> list[str]:
    out: list[str] = []
    for s in (chain or []):
        if not isinstance(s, dict):
            continue
        rel = str(s.get("rel") or s.get("rel_name") or "").strip()
        if rel:
            out.append(rel)
            continue
        legacy = str(s.get("join_field") or s.get("jf") or "").strip()
        if legacy:
            seg = jf_short(legacy)
            if seg:
                out.append(seg)
    return out


_PROP_REF_RE = re.compile(r"(?:has|values)\(\s*['\"]([A-Za-z_]\w*)['\"]")
_AGG_METHOD_RE = re.compile(r"\.(sum|mean|min|max|count)\s*\(\s*\)")


def _agg_from_plan(plan) -> str:
    if not isinstance(plan, dict) or not plan.get("steps"):
        return ""
    final = plan.get("final") if isinstance(plan.get("final"), dict) else {}
    sid = str(final.get("step") or "").strip()
    text = ""
    if sid:
        for st in plan.get("steps") or []:
            if isinstance(st, dict) and str(st.get("id") or "") == sid:
                text = str(st.get("gremlin") or "")
                break
    if not text:
        text = json.dumps(plan, ensure_ascii=False, default=str)
    hits = _AGG_METHOD_RE.findall(text)
    return hits[-1].lower() if hits else ""


def final_agg_method(f) -> str:
    return _agg_from_plan(get_field(f, "gremlin_plan", None))


def _plan_signals(f) -> tuple:
    plan = get_field(f, "gremlin_plan", None)
    if not isinstance(plan, dict) or not plan.get("steps"):
        return ((), "")
    text = json.dumps(plan, ensure_ascii=False, default=str)
    attrs = tuple(sorted(set(_PROP_REF_RE.findall(text))))
    final = plan.get("final") if isinstance(plan.get("final"), dict) else {}
    stat = str(final.get("op") or "").strip().lower()
    if not stat:
        stat = _agg_from_plan(plan) or str(final.get("type") or "").strip().lower()
    return (attrs, stat)


def structure_signature(f) -> tuple:
    chain = get_field(f, "evidence_chain", None) or []
    rels = tuple(rel_short(chain))
    terminal = ""
    for s in reversed(chain):
        if not isinstance(s, dict):
            continue
        jf = str(s.get("join_field") or s.get("jf") or "").strip()
        if not jf:
            continue
        tail = jf.partition("|")[2].strip()
        last = tail.split("&")[-1].strip() if tail else ""
        if last in ("payer", "payee"):
            terminal = "account"
        elif last:
            terminal = last
        if terminal:
            break
    evt = str(get_field(f, "expected_value_type", "") or "").strip()
    return (rels, terminal, evt, _plan_signals(f))


def candidate_similarity(a, b) -> float:
    from . import paradigm
    sa, sb = structure_signature(a), structure_signature(b)
    if not sa[0] or not sb[0]:
        return 0.2
    fourth_known = bool(sa[3][0] or sa[3][1]) and bool(sb[3][0] or sb[3][1])
    if sa[:3] == sb[:3] and (not fourth_known or sa[3] == sb[3]):
        return 0.9
    if sa[0] == sb[0]:
        return 0.65
    if paradigm.same_paradigm(a, b):
        return 0.5
    return 0.2
