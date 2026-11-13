from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from ..utils import get_logger


logger = get_logger("execution.plan")
HUB_VLABEL = "entity"
HUB_ELABEL = "relation"


_SEED_PH_RE = re.compile(r"\{\{SEED\}\}")

_STEP_ID_PATTERN = r"s[A-Za-z0-9_]*"
_STEP_PH_RE = re.compile(r"\{\{STEP:(" + _STEP_ID_PATTERN + r")\}\}")


_QUOTED_SEED_PH_RE = re.compile(r"['\"](\{\{SEED\}\})['\"]")
_QUOTED_STEP_PH_RE = re.compile(r"['\"](\{\{STEP:" + _STEP_ID_PATTERN + r"\}\})['\"]")
_ANY_STEP_PH_RE = re.compile(r"\{\{STEP:([^}]*)\}\}")
_EDGE_INT_PROPS = frozenset({"total_amount", "trade_count"})
_AMOUNT_AGG_RE = re.compile(
    r"values\s*\(\s*['\"](total_amount)['\"]\s*\)\s*\.\s*(sum|mean|avg|min|max|fold)\s*\(",
    re.IGNORECASE)
_QUOTED_NUM_RE = re.compile(r"^['\"](-?\d+(?:\.\d+)?)['\"]$")
_HAS_NUM_PRED_RE = re.compile(
    r"(has\s*\(\s*['\"])([\w.|$]+)(['\"]\s*,\s*)"
    r"(gte|gt|lte|lt|eq|neq|between|within)\s*\(([^()]*)\)(\s*\))",
    re.IGNORECASE)
_HAS_NUM_EQ_RE = re.compile(
    r"(has\s*\(\s*['\"])([\w.|$]+)(['\"]\s*,\s*)(-?\d+(?:\.\d+)?)(\s*\))")
ID_LIST_LIMIT = 300
_EMPTY_SRC_PATTERNS = (
    re.compile(r"^\s*g_[A-Za-z0-9_]+\.V\(\s*\[\s*\]\s*\)"),
    re.compile(r"^\s*g_[A-Za-z0-9_]+\.V\(\s*\)\.hasId\(\s*"
               r"(?:within\s*\(\s*)?\[\s*\]"),
    re.compile(r"^\s*g_[A-Za-z0-9_]+\.V\(\s*within\s*\(\s*\[\s*\]"),
)
_QUERY_MODE = "binding"


def set_query_mode(mode: str) -> str:
    global _QUERY_MODE
    m = str(mode or "").strip().lower()
    if m not in ("binding", "inline"):
        logger.warning(f"[query mode] Invalid value {mode!r}; falling back to binding")
        m = "binding"
    if m != _QUERY_MODE:
        logger.info(f"[query mode] {_QUERY_MODE} -> {m}")
    _QUERY_MODE = m
    return _QUERY_MODE


def query_mode() -> str:
    return _QUERY_MODE


def _trunc(s: Any, n: int = 300) -> str:
    t = str(s)
    return t if len(t) <= n else t[:n] + f"...(+{len(t) - n} chars)"


VID_SIZE = 128
_VID_TRUNCATIONS: Dict[str, str] = {}


def sanitize_vid(vid: Any) -> str:
    if vid is None:
        return ""
    s = str(vid)
    encoded = s.encode("utf-8")
    if len(encoded) <= VID_SIZE:
        return s
    truncated = encoded[:VID_SIZE].decode("utf-8", errors="ignore")
    if s not in _VID_TRUNCATIONS:
        _VID_TRUNCATIONS[s] = truncated
        logger.warning(f"[vid truncation] Business ID exceeds {VID_SIZE}B; truncated to match ingestion "
                       f"(total {len(_VID_TRUNCATIONS)}): ...{truncated[-40:]}")
    return truncated


def vid_truncation_stats() -> int:
    return len(_VID_TRUNCATIONS)


def vid_to_token(vid: str) -> str:
    v = str(vid or "").strip()
    if not v:
        return v
    if "|" in v:
        biz, _, lbl = v.rpartition("|")
        if biz:
            return f"{sanitize_vid(biz)}|{lbl}"
        return v
    return f"{sanitize_vid(v)}|{HUB_VLABEL}"


def _topo_order(dep_map: Dict[str, List[str]]) -> Optional[List[List[str]]]:
    remaining = {k: set(v) for k, v in dep_map.items()}
    done: Set[str] = set()
    levels: List[List[str]] = []
    while remaining:
        ready = sorted(k for k, deps in remaining.items() if deps <= done)
        if not ready:
            return None
        levels.append(ready)
        for k in ready:
            done.add(k)
            remaining.pop(k, None)
    return levels


def normalize_plan(obj: Dict[str, Any]) -> Dict[str, Any]:
    steps_in = obj.get("steps") or []
    norm_steps: List[Dict[str, Any]] = []
    for i, st in enumerate(steps_in):
        sid = str(st.get("id") or f"s{i + 1}")
        gremlin = str(st.get("gremlin") or "").strip()
        deps = list(dict.fromkeys(
            [str(d) for d in (st.get("depends_on") or [])] + _STEP_PH_RE.findall(gremlin)))
        norm_steps.append({
            "id": sid,
            "desc": str(st.get("desc") or "").strip(),
            "depends_on": deps,
            "output": str(st.get("output") or "id_list"),
            "gremlin": gremlin,
        })
    final = obj.get("final")
    return {
        "feature_understanding": str(obj.get("feature_understanding") or "").strip(),
        "steps": norm_steps,
        "final": final if isinstance(final, dict) else {},
        "business_rationale": str(obj.get("business_rationale") or "").strip(),
    }


def _str_lit(num_src: str) -> str:
    f = float(num_src)
    return f"'{int(f)}'" if f.is_integer() else f"'{f}'"


def _coerce_numeric_literals(gremlin: str) -> str:
    if not gremlin or not re.search(r"\d", gremlin):
        return gremlin

    def _fix_args(args_src: str) -> str:
        out = []
        for part in args_src.split(","):
            value = part.strip()
            if value and not value.startswith(("'", '"')) and re.fullmatch(r"-?\d+(\.\d+)?", value):
                part = _str_lit(value)
            out.append(part)
        return ",".join(out)

    def _repl_pred(m: "re.Match[str]") -> str:
        prop = m.group(2)
        if prop.lower() in _EDGE_INT_PROPS:
            return m.group(0)
        args = m.group(5)
        if not re.search(r"(?<![\d.'\"])\d+(?:\.\d+)?(?![\d.'\"])", args):
            return m.group(0)
        return (m.group(1) + prop + m.group(3) + m.group(4)
                + "(" + _fix_args(args) + ")" + m.group(6))

    def _repl_eq(m: "re.Match[str]") -> str:
        prop = m.group(2)
        if prop.lower() in _EDGE_INT_PROPS:
            return m.group(0)
        return m.group(1) + prop + m.group(3) + _str_lit(m.group(4)) + m.group(5)

    out = _HAS_NUM_PRED_RE.sub(_repl_pred, gremlin)
    return _HAS_NUM_EQ_RE.sub(_repl_eq, out)


def sanitize_plan(plan: Dict[str, Any], seed_vid: str, g_prefix: str) -> List[str]:
    fixes: List[str] = []
    seed = str(seed_vid or "").strip()
    token = vid_to_token(seed) if seed else ""
    g_source_pat = re.compile(re.escape(g_prefix) + r"\.V\(")
    for st in plan.get("steps") or []:
        g = st.get("gremlin") or ""
        if not g:
            continue
        orig = g
        if token:
            for lit in dict.fromkeys((token, seed)):
                if not lit:
                    continue
                pat = re.compile(
                    r"(?<![\w|])['\"]" + re.escape(lit) + r"['\"](?![\w|])"
                    r"|(?<![\w|'\"])" + re.escape(lit) + r"(?![\w|])")
                g = pat.sub("{{SEED}}", g)
        g = re.sub(r"\.toList(?!\()", ".toList()", g)
        if "__G__" in g:
            g = g.replace("__G__", g_prefix)
            fixes.append(f"__G__ placeholder -> {g_prefix}")
        g = re.sub(r"(?<![\w.])g\.(?=[A-Za-z])", lambda m: f"{g_prefix}.", g)
        g = g_source_pat.sub(lambda m: m.group(0) if m.start() == 0 else "__.V(", g)
        g = re.sub(r"\.asDouble\(\)", "", g)
        g2 = _coerce_numeric_literals(g)
        if g2 != g:
            fixes.append(f"{st.get('id')}: numeric literals coerced by property class "
                         f"(edge integer/vertex property string): "
                         f"{_trunc(g, 90)} -> {_trunc(g2, 90)}")
            g = g2
        if g != orig:
            st["gremlin"] = g
            fixes.append(f"{st.get('id')}: {_trunc(orig, 90)} -> {_trunc(g, 90)}")
    return fixes


def validate_plan(plan: Dict[str, Any]) -> Tuple[bool, str]:
    steps = plan.get("steps") or []
    if not steps:
        return False, "the plan has no steps"
    ids: List[str] = []
    id2out: Dict[str, str] = {}
    for st in steps:
        sid = st.get("id")
        if not sid:
            return False, "a step is missing its id"
        if not re.fullmatch(_STEP_ID_PATTERN, sid):
            return False, (f"invalid step id: {sid} (must start with s and contain only letters, digits, or underscores, "
                           f"e.g. s1_out_amt)")
        if sid in id2out:
            return False, f"duplicate step id: {sid}"
        if st.get("output") not in ("id_list", "scalar", "scalar_list"):
            return False, f"step {sid} has invalid output: {st.get('output')} (allowed: id_list/scalar/scalar_list)"
        if not st.get("gremlin"):
            return False, f"step {sid} has an empty gremlin"
        ids.append(sid)
        id2out[sid] = st["output"]
    id_set = set(ids)
    dep_map: Dict[str, List[str]] = {}
    for st in steps:
        sid = st["id"]
        for d in (st.get("depends_on") or []):
            if d not in id_set:
                return False, f"step {sid} depends on undefined step {d}"
            if d == sid:
                return False, f"step {sid} depends on itself"
        dep_map[sid] = list(st.get("depends_on") or [])
    if _topo_order(dep_map) is None:
        return False, "step dependencies contain a cycle"
    for st in steps:
        sid = st["id"]
        g = st["gremlin"]
        deps = set(st.get("depends_on") or [])
        for raw in set(_ANY_STEP_PH_RE.findall(g)):
            if not re.fullmatch(_STEP_ID_PATTERN, raw):
                return False, (f"step {sid} uses an invalid placeholder {{{{STEP:{raw}}}}}; "
                               f"step ids must start with s and contain only letters, digits, or underscores; "
                               f"only defined step ids ({'/'.join(ids)}) are allowed")
            if raw not in id_set:
                return False, (f"step {sid} references undefined {{{{STEP:{raw}}}}}; "
                               f"defined step ids are ({'/'.join(ids)}) and references must match exactly")
        step_refs = set(_STEP_PH_RE.findall(g))
        for r in step_refs:
            if r not in deps:
                return False, f"step {sid} references {{{{STEP:{r}}}}} but omits it from depends_on"
            if id2out.get(r) != "id_list":
                return False, f"step {sid} references {r}, whose output is not id_list and cannot seed a traversal"
        m_empty = re.search(r"\.(or|and|not)\s*\(\s*\)", g)
        if m_empty:
            return False, (f"step {sid} contains an empty call .{m_empty.group(1)}(); or()/and()/not() "
                           f"require anonymous traversal arguments, e.g. .or(__.has('join_field','a'), __.has('join_field','b'))")
        has_seed = bool(_SEED_PH_RE.search(g))
        if not has_seed and not step_refs:
            return False, f"step {sid} neither anchors on {{{{SEED}}}} nor consumes {{{{STEP:sX}}}}; this may scan the whole graph"
        if re.search(r"\.V\(\s*\)", g) and not step_refs and not has_seed:
            return False, f"step {sid} uses an unanchored V(); this may scan the whole graph"
    ok, err = _validate_final(plan.get("final") or {}, id2out)
    if not ok:
        return False, err
    return True, ""


def _validate_final(final: Dict[str, Any], id2out: Dict[str, str]) -> Tuple[bool, str]:
    t = final.get("type")
    if t == "scalar":
        s = final.get("step")
        if s not in id2out:
            return False, f"final.scalar references undefined step {s}"
        if id2out[s] != "scalar":
            return False, f"final.scalar step {s} does not output scalar"
        return True, ""
    if t == "count":
        s = final.get("step")
        if s not in id2out:
            return False, f"final.count references undefined step {s}"
        if id2out[s] != "id_list":
            return False, f"final.count step {s} does not output id_list"
        return True, ""
    if t == "exists":
        s = final.get("step")
        if s not in id2out:
            return False, f"final.exists references undefined step {s}"
        return True, ""
    if t == "ratio":
        for key, mode_key in (("numerator_step", "numerator_mode"),
                              ("denominator_step", "denominator_mode")):
            s = final.get(key)
            mode = final.get(mode_key)
            if s not in id2out:
                return False, f"final.ratio references undefined step {s}"
            if mode not in ("count", "scalar"):
                return False, f"final.ratio has invalid {mode_key}: {mode}"
            need = "id_list" if mode == "count" else "scalar"
            if id2out[s] != need:
                return False, f"final.ratio step {s} must output {need} (mode={mode})"
        return True, ""
    if t == "reduce":
        s = final.get("step")
        if s not in id2out:
            return False, f"final.reduce references undefined step {s}"
        if id2out[s] != "scalar_list":
            return False, f"final.reduce step {s} does not output scalar_list"
        if final.get("op") not in ("sum", "min", "max", "avg", "mean"):
            return False, f"final.reduce has invalid op: {final.get('op')}"
        return True, ""
    return False, f"final.type is invalid or missing: {t} (allowed: scalar/count/exists/reduce/ratio)"


def _split_top_commas(s: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    in_str = False
    esc = False
    quote = ""
    cur: List[str] = []
    for ch in s:
        if in_str:
            cur.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch in ("'", '"'):
            in_str = True
            quote = ch
            cur.append(ch)
        elif ch in "([":
            depth += 1
            cur.append(ch)
        elif ch in ")]":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]


def _flatten_pred_list_args(gremlin: str) -> str:
    pat = re.compile(r"(?:within|hasId)\(")
    is_list = lambda p: p.startswith("[") and p.endswith("]")
    is_scalar = lambda p: bool(re.fullmatch(r"(['\"]).*\1|-?\d+(\.\d+)?", p, re.DOTALL))

    def _try_merge_once(s: str) -> Tuple[str, bool]:
        for m in pat.finditer(s):
            depth = 1
            in_str = False
            esc = False
            quote = ""
            j = m.end()
            while j < len(s) and depth:
                ch = s[j]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == quote:
                        in_str = False
                else:
                    if ch in ("'", '"'):
                        in_str = True
                        quote = ch
                    elif ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            break
                j += 1
            if depth:
                return s, False
            inner = s[m.end():j]
            parts = _split_top_commas(inner)
            if len(parts) >= 2 and all(is_list(p) or is_scalar(p) for p in parts):
                elems: List[str] = []
                for p in parts:
                    if is_list(p):
                        elems.extend(_split_top_commas(p[1:-1]))
                    else:
                        elems.append(p)
                merged = "[" + ", ".join(elems) + "]"
                return s[:m.end()] + merged + s[j:], True
        return s, False

    for _ in range(8):
        gremlin, changed = _try_merge_once(gremlin)
        if not changed:
            break
    return gremlin


_SEED_BIND_NAME = "s"
_STEP_BIND_PREFIX = "l_"
_MERGED_BIND_PREFIX = "lm"

_PRED_CALL_RE = re.compile(r"(?:within|hasId)\s*\(")

_STEP_PH_ONLY_RE = re.compile(r"^\{\{STEP:(" + _STEP_ID_PATTERN + r")\}\}$")
_HEAD_V_WITHIN_RE = re.compile(
    r"^(?P<indent>\s*)(?P<g>g_[A-Za-z0-9_]+)\.V\(\)\.hasId\(within\("
    r"(?P<var>[A-Za-z_][A-Za-z0-9_]*)\)\)")
_HEAD_V_BARE_RE = re.compile(
    r"^(?P<indent>\s*)(?P<g>g_[A-Za-z0-9_]+)\.V\(\)\.hasId\("
    r"(?P<var>[A-Za-z_][A-Za-z0-9_]*)\)")


def _paren_end(s: str, open_idx: int) -> int:
    depth = 0
    in_str = False
    esc = False
    quote = ""
    j = open_idx
    while j < len(s):
        ch = s[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
        else:
            if ch in ("'", '"'):
                in_str = True
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return j
        j += 1
    return -1


def _merge_multi_step_ph(gremlin: str, inter: Dict[str, List[Any]],
                         bindings: Dict[str, Any]) -> str:
    merged: Dict[Tuple[str, ...], str] = {}
    edits: List[Tuple[int, int, str]] = []
    for m in _PRED_CALL_RE.finditer(gremlin):
        j = _paren_end(gremlin, m.end() - 1)
        if j < 0:
            continue
        sids: List[str] = []
        for p in _split_top_commas(gremlin[m.end():j]):
            mm = _STEP_PH_ONLY_RE.match(p.strip())
            if not mm:
                sids = []
                break
            sids.append(mm.group(1))
        if not sids:
            continue
        if len(sids) == 1:
            var = f"{_STEP_BIND_PREFIX}{sids[0]}"
        else:
            key = tuple(sids)
            if key not in merged:
                merged[key] = f"{_MERGED_BIND_PREFIX}{len(merged) + 1}"
            var = merged[key]
        if var not in bindings:
            vals: List[str] = []
            for sid in sids:
                vals.extend(vid_to_token(str(x)) for x in (inter.get(sid) or []))
            bindings[var] = vals
        edits.append((m.end(), j, var))
    for st, en, var in reversed(edits):
        gremlin = gremlin[:st] + var + gremlin[en:]
    return gremlin


def _normalize_head_v(gremlin: str) -> str:
    out = _HEAD_V_WITHIN_RE.sub(
        lambda m: f"{m.group('indent')}{m.group('g')}.V({m.group('var')})", gremlin)
    return _HEAD_V_BARE_RE.sub(
        lambda m: f"{m.group('indent')}{m.group('g')}.V({m.group('var')})", out)


_LIT_TOKEN_RE = re.compile(
    r"^(?:'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|-?\d+(?:\.\d+)?)$")

_IDENT_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_WITHIN_HEAD_RE = re.compile(r"\bwithin\s*\(")
_HAS_HEAD_RE = re.compile(r"\bhas\s*\(|\bhasId\s*\(")
_SCALAR_PRED_HEAD_RE = re.compile(
    r"\b(?:eq|neq|gte|gt|lte|lt|between|inside|outside)\s*\(")
_JF_NEAR_RE = re.compile(r"join_field['\"]\s*,\s*$")


def _unquote(tok: str) -> Any:
    t = tok.strip()
    if len(t) >= 2 and t[0] in ("'", '"') and t[-1] == t[0]:
        return re.sub(r"\\(.)", r"\1", t[1:-1])
    return float(t) if "." in t else int(t)


def _literalize_bindings(gremlin: str, bindings: Dict[str, Any]) -> str:
    counters = {"jf": 0, "wl": 0, "v": 0}
    named: Dict[Tuple[str, str, Any], str] = {}
    edits: List[Tuple[int, int, str]] = []

    def _list_var(kind: str, values: List[Any]) -> str:
        key = (kind, "list", tuple(values))
        if key not in named:
            counters[kind] += 1
            named[key] = f"{kind}{counters[kind]}"
        var = named[key]
        if var not in bindings:
            bindings[var] = list(values)
        return var

    def _scalar_var(val: Any) -> str:
        key = ("v", type(val).__name__, val)
        if key not in named:
            counters["v"] += 1
            named[key] = f"v{counters['v']}"
        var = named[key]
        if var not in bindings:
            bindings[var] = val
        return var

    for m in _HAS_HEAD_RE.finditer(gremlin):
        j = _paren_end(gremlin, m.end() - 1)
        if j < 0:
            continue
        args = _split_top_commas(gremlin[m.end():j])
        if m.group(0).startswith("hasId"):
            if len(args) == 1 and _LIT_TOKEN_RE.match(args[0].strip()):
                edits.append((m.end(), j, _scalar_var(_unquote(args[0]))))
            continue
        if len(args) != 2:
            continue
        prop_tok = args[0].strip()
        if not (len(prop_tok) >= 2 and prop_tok[0] in ("'", '"')):
            continue
        val_tok = args[1].strip()
        if _LIT_TOKEN_RE.match(val_tok):
            edits.append((m.end(), j, f"{args[0]}, {_scalar_var(_unquote(val_tok))}"))
    for m in _WITHIN_HEAD_RE.finditer(gremlin):
        j = _paren_end(gremlin, m.end() - 1)
        if j < 0:
            continue
        parts = [p.strip() for p in _split_top_commas(gremlin[m.end():j])]
        if not parts or any(not p for p in parts):
            continue
        vals: List[Any] = []
        n_ident = 0
        ok = True
        for p in parts:
            if _LIT_TOKEN_RE.match(p):
                vals.append(_unquote(p))
            elif _IDENT_TOKEN_RE.match(p) and p in bindings:
                n_ident += 1
                bv = bindings[p]
                vals.extend(bv if isinstance(bv, list) else [bv])
            else:
                ok = False
                break
        if not ok:
            continue
        if n_ident == 1 and len(parts) == 1:
            continue
        prefix = gremlin[max(0, m.start() - 64):m.start()]
        kind = "jf" if _JF_NEAR_RE.search(prefix) else "wl"
        edits.append((m.end(), j, _list_var(kind, vals)))
    for m in _SCALAR_PRED_HEAD_RE.finditer(gremlin):
        j = _paren_end(gremlin, m.end() - 1)
        if j < 0:
            continue
        parts = _split_top_commas(gremlin[m.end():j])
        if not parts or any(not p.strip() for p in parts):
            continue
        new_args: List[str] = []
        changed = False
        for p in parts:
            t = p.strip()
            if _LIT_TOKEN_RE.match(t):
                new_args.append(_scalar_var(_unquote(t)))
                changed = True
            else:
                new_args.append(p)
        if changed:
            edits.append((m.end(), j, ", ".join(new_args)))
    for st, en, var in sorted(edits, key=lambda e: e[0], reverse=True):
        gremlin = gremlin[:st] + var + gremlin[en:]
    return gremlin


def _apply_bindings(gremlin: str, seed_vid: str,
                    inter: Dict[str, List[Any]]) -> Tuple[str, Dict[str, Any], bool]:
    bindings: Dict[str, Any] = {}
    out = _merge_multi_step_ph(gremlin, inter, bindings)
    if _SEED_PH_RE.search(out):
        out = _SEED_PH_RE.sub(_SEED_BIND_NAME, out)
        bindings[_SEED_BIND_NAME] = vid_to_token(seed_vid)

    def _rep(m: "re.Match[str]") -> str:
        sid = m.group(1)
        name = f"{_STEP_BIND_PREFIX}{sid}"
        if name not in bindings:
            bindings[name] = [vid_to_token(str(x)) for x in (inter.get(sid) or [])]
        return name

    out = _STEP_PH_RE.sub(_rep, out)
    out = _normalize_head_v(out)
    out = _literalize_bindings(out, bindings)
    out = _normalize_head_v(out)
    empty = any(isinstance(v, list) and not v for v in bindings.values())
    return out, bindings, empty


def _apply_inline(gremlin: str, seed_vid: str,
                  inter: Dict[str, List[Any]]) -> Tuple[str, Dict[str, Any], bool]:
    empty = False
    out = _SEED_PH_RE.sub(json.dumps(vid_to_token(seed_vid), ensure_ascii=False), gremlin)

    def _rep(m: "re.Match[str]") -> str:
        nonlocal empty
        sid = m.group(1)
        ids = inter.get(sid) or []
        tokens = [vid_to_token(str(x)) for x in ids]
        if not tokens:
            empty = True
        return "[" + ", ".join(json.dumps(t, ensure_ascii=False) for t in tokens) + "]"

    out = _STEP_PH_RE.sub(_rep, out)
    out = _flatten_pred_list_args(out)
    return out, {}, empty


def _resolve_step_gremlin(gremlin: str, seed_vid: str,
                          inter: Dict[str, List[Any]]
                          ) -> Tuple[str, Dict[str, Any], bool]:
    gremlin = _QUOTED_SEED_PH_RE.sub(r"\1", gremlin)
    gremlin = _QUOTED_STEP_PH_RE.sub(r"\1", gremlin)
    if _QUERY_MODE == "binding":
        return _apply_bindings(gremlin, seed_vid, inter)
    return _apply_inline(gremlin, seed_vid, inter)


def _bindings_digest(bindings: Dict[str, Any], max_len: int = 300) -> str:
    if not bindings:
        return ""
    parts: List[str] = []
    for k, v in bindings.items():
        if isinstance(v, list):
            if not v:
                parts.append(f"{k}=[] (empty)")
            else:
                sample = ", ".join(str(x)[:44] for x in v[:2])
                more = f" ... {len(v)} total" if len(v) > 2 else ""
                parts.append(f"{k}=[{len(v)} items] {sample}{more}")
        else:
            if isinstance(v, str):
                parts.append(f'{k}="{v[:80]}"')
            else:
                parts.append(f"{k}={v!r}")
    s = "; ".join(parts)
    return s if len(s) <= max_len else s[:max_len] + "…"


def _final_reduce(final: Dict[str, Any], inter: Dict[str, List[Any]],
                  scalars: Dict[str, Any], scalar_lists: Dict[str, List[Any]],
                  note: Optional[Dict[str, Any]] = None) -> Any:
    t = final.get("type")
    step = final.get("step")
    if t == "scalar":
        return scalars.get(step) if isinstance(step, str) else None
    if t == "count":
        return len(inter.get(step) or []) if isinstance(step, str) else 0
    if t == "exists":
        return 1 if (isinstance(step, str) and (inter.get(step) or [])) else 0
    if t == "reduce":
        op = str(final.get("op") or "sum").lower()
        vals = [float(v) for v in ((scalar_lists.get(step) or []) if isinstance(step, str) else [])
                if isinstance(v, (int, float))]
        if not vals:
            return 0.0
        if op == "sum":
            return sum(vals)
        if op in ("avg", "mean"):
            return sum(vals) / len(vals)
        if op == "min":
            return min(vals)
        if op == "max":
            return max(vals)
        return sum(vals)
    if t == "ratio":
        def _val(step_name: str, mode: str) -> float:
            if mode == "count":
                return float(len(inter.get(step_name) or []))
            v = scalars.get(step_name)
            return float(v) if isinstance(v, (int, float)) else 0.0
        num_step = final.get("numerator_step")
        num_mode = final.get("numerator_mode")
        den_step = final.get("denominator_step")
        den_mode = final.get("denominator_mode")
        if not (isinstance(num_step, str) and isinstance(num_mode, str)
                and isinstance(den_step, str) and isinstance(den_mode, str)):
            return 0.0
        num = _val(num_step, num_mode)
        den = _val(den_step, den_mode)
        if not den:
            if note is not None:
                note["ratio_den_zero"] = {"numerator": num,
                                          "denominator_step": den_step,
                                          "denominator_mode": den_mode}
            return 0.0
        return num / den
    return None


SERVER_OUTAGE_ERR_PREFIX = "server unavailable"
SLOW_QUERY_ERR_PREFIX = "slow query/timeout"


def execute_plan(executor: Any, plan: Dict[str, Any], seed_vid: str,
                 *, no_exec: bool = False, step_timeout_s: float = 30.0,
                 transient_retries: int = 3,
                 transient_backoff_s: float = 1.0) -> Tuple[Any, List[Dict[str, Any]], str]:
    _log = logger.info
    from .eg_client import is_slow_query_error, is_transient_exec_error

    local_execute = getattr(executor, "execute_plan_local", None)
    if not no_exec and callable(local_execute):
        return local_execute(plan, seed_vid)
    steps = plan.get("steps") or []
    dep_map = {st["id"]: list(st.get("depends_on") or []) for st in steps}
    levels = _topo_order(dep_map)
    if levels is None:
        return None, [], "step dependencies contain a cycle"
    inter: Dict[str, List[Any]] = {}
    scalars: Dict[str, Any] = {}
    scalar_lists: Dict[str, List[Any]] = {}
    trace: List[Dict[str, Any]] = []
    by_id = {st["id"]: st for st in steps}
    for level in levels:
        for sid in level:
            st = by_id[sid]
            resolved0, bindings, empty_input = _resolve_step_gremlin(
                st["gremlin"], seed_vid, inter)
            out_type = st.get("output")
            is_amt_agg = bool(_AMOUNT_AGG_RE.search(resolved0))
            resolved = resolved0
            rec: Dict[str, Any] = {"id": sid, "desc": st.get("desc"), "output": out_type,
                                   "resolved_gremlin": resolved}
            if is_amt_agg:
                rec["amt_unit"] = "total_amount values and aggregates are integer fen without conversion"
            if bindings:
                rec["query_mode"] = "binding"
                rec["bindings_digest"] = _bindings_digest(bindings)
            _log(f"    [step {sid}] output={out_type} desc={st.get('desc')}")
            _log(f"        gremlin: {_trunc(resolved, 260)}")
            if bindings:
                _log(f"        bindings: {_trunc(rec['bindings_digest'], 200)}")
            if no_exec:
                rec["result_preview"] = "(not executed)"
                trace.append(rec)
                continue
            if empty_input or any(p.match(resolved) for p in _EMPTY_SRC_PATTERNS):
                if out_type == "id_list":
                    inter[sid] = []
                elif out_type == "scalar":
                    scalars[sid] = 0
                else:
                    scalar_lists[sid] = []
                rec["result_preview"] = ("0 (short-circuited: an upstream hop is empty, no request sent)"
                                         if out_type != "scalar_list"
                                         else "0 values (short-circuited: empty input)")
                rec["short_circuit"] = True
                trace.append(rec)
                _log(f"        -> {rec['result_preview']}")
                continue
            t0 = time.time()
            rows: List[Any] = []
            exec_exc: Optional[Exception] = None
            try_i = 0
            for try_i in range(transient_retries + 1):
                try:
                    if executor is None:
                        raise RuntimeError("executor is not initialized")
                    rows = executor.submit(resolved, bindings)
                    exec_exc = None
                    break
                except Exception as e:
                    exec_exc = e
                    if is_slow_query_error(e):
                        break
                    if try_i < transient_retries and is_transient_exec_error(e):
                        wait_s = transient_backoff_s * (try_i + 1)
                        _log(f"        -> Request failed ({_trunc(str(e), 140)}); "
                             f"retry in {wait_s:.0f}s ({try_i + 1}/{transient_retries})")
                        time.sleep(wait_s)
                        if executor is not None:
                            try:
                                executor.reset()
                            except Exception:
                                pass
                        continue
                    break
            if exec_exc is not None:
                rec["error"] = f"{type(exec_exc).__name__}: {exec_exc}"
                if try_i > 0:
                    rec["retries"] = try_i
                trace.append(rec)
                if is_slow_query_error(exec_exc):
                    return None, trace, (
                        f"{SLOW_QUERY_ERR_PREFIX}: step {sid} exceeded the server time budget; "
                        f"treated as failure; last error: "
                        f"{type(exec_exc).__name__}: {exec_exc}")
                if is_transient_exec_error(exec_exc):
                    return None, trace, (
                        f"{SERVER_OUTAGE_ERR_PREFIX}: step {sid} failed after "
                        f"{transient_retries} retries; "
                        f"last error: {type(exec_exc).__name__}: {exec_exc}")
                return None, trace, f"step {sid} execution failed: {type(exec_exc).__name__}: {exec_exc}"
            if try_i > 0:
                rec["retries"] = try_i
            dt = time.time() - t0
            rec["elapsed_s"] = round(dt, 2)
            if dt > step_timeout_s:
                rec["error"] = f"timeout {dt:.1f}s"
                trace.append(rec)
                return None, trace, f"step {sid} timed out after {dt:.1f}s"
            if out_type == "id_list":
                ids = [str(x) for x in rows if x is not None][:ID_LIST_LIMIT]
                inter[sid] = ids
                rec["result_preview"] = f"{len(ids)} ids" + (" (truncated)" if len(rows) > ID_LIST_LIMIT else "")
            elif out_type == "scalar":
                val = rows[0] if rows else None
                if isinstance(val, dict):
                    for v in val.values():
                        if isinstance(v, (int, float)):
                            val = v
                            break
                scalars[sid] = val
                rec["result_preview"] = f"scalar={val}" + (" fen" if is_amt_agg else "")
            elif out_type == "scalar_list":
                vals = [x for x in rows if isinstance(x, (int, float))]
                scalar_lists[sid] = vals
                rec["result_preview"] = f"{len(vals)} values" + (" (fen)" if is_amt_agg else "")
            trace.append(rec)
            _log(f"        -> {rec['result_preview']} elapsed {dt:.2f}s")

    note: Dict[str, Any] = {}
    value = _final_reduce(plan.get("final") or {}, inter, scalars, scalar_lists, note)
    _log(f"    [final] {json.dumps(plan.get('final'), ensure_ascii=False)} -> value={value!r}")
    if note:
        _log(f"        -> reduction notes: {json.dumps(note, ensure_ascii=False)}")
        trace.append({"id": "final", "output": "final",
                      "desc": "final reduction (ratio denominator is empty; value set to 0)",
                      "result_preview": f"value={value!r}", "note": note})
    return value, trace, ""


def render_execution_trace(execution_trace: Optional[List[Dict[str, Any]]]) -> str:
    if not execution_trace:
        return ""
    lines: List[str] = []
    for t in execution_trace:
        sid = t.get("id", "")
        desc = t.get("desc", "")
        out = t.get("output", "")
        preview = t.get("result_preview", "")
        resolved = t.get("resolved_gremlin", "")
        err = t.get("error")
        lines.append(f"-- [{sid}] ({out}) {desc}  ==> {preview}" + (f"  [ERR {err}]" if err else ""))
        if resolved:
            lines.append(f"   {resolved}")
        bd = t.get("bindings_digest")
        if bd:
            lines.append(f"   [bindings] {bd}")
    return "\n".join(lines)
