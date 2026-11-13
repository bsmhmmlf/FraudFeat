# -*- coding: utf-8 -*-
"""Deterministic local engine for the constrained FraudFeat Gremlin subset.

Only operators listed in ``SUPPORTED`` are accepted; unsupported or ambiguous
operations fail explicitly instead of returning an approximate value. ``{{SEED}}``
and ``{{STEP:sX}}`` placeholders are expanded locally. Vertex properties use string
comparison, while numeric semantics apply to ``total_amount``, ``trade_count``, and
``top_k`` edge attributes.

Released amounts are stored in fen. The default ``amount_unit='fen'`` keeps amount
thresholds and aggregates in that unit, matching the EvoFeat synthesizer.
``amount_unit='yuan'`` multiplies thresholds by 100 and divides aggregates by 100.

Public entry point::

    value, trace, error = execute_plan_local(store, plan, seed_unique_id)
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .graph_store import GraphStore


__all__ = [
    "EngineError", "SUPPORTED", "execute_plan_local", "normalize_plan",
    "validate_plan", "is_zero_hit_value", "ratio_den_zero_of",
]

_STEP_ID_PATTERN = r"s[A-Za-z0-9_]*"
_STEP_PH_RE = re.compile(r"\{\{STEP:(" + _STEP_ID_PATTERN + r")\}\}")
_SEED_PH_RE = re.compile(r"\{\{SEED\}\}")
_ANY_STEP_PH_RE = re.compile(r"\{\{STEP:([^}]*)\}\}")
_QUOTED_SEED_PH_RE = re.compile(r"['\"](\{\{SEED\}\})['\"]")
_QUOTED_STEP_PH_RE = re.compile(r"['\"](\{\{STEP:" + _STEP_ID_PATTERN + r"\}\})['\"]")


def _topo_order(dep_map: Dict[str, List[str]]) -> Optional[List[List[str]]]:
    remaining = {k: set(v) for k, v in dep_map.items()}
    done: set = set()
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


def _validate_final(final: Dict[str, Any], id2out: Dict[str, str]) -> Tuple[bool, str]:
    t = final.get("type")
    if t == "scalar":
        s = final.get("step")
        if s not in id2out:
            return False, f"final.scalar references unknown step {s}"
        if id2out[s] != "scalar":
            return False, f"final.scalar step {s} must produce scalar"
        return True, ""
    if t == "count":
        s = final.get("step")
        if s not in id2out:
            return False, f"final.count references unknown step {s}"
        if id2out[s] != "id_list":
            return False, f"final.count step {s} must produce id_list"
        return True, ""
    if t == "exists":
        s = final.get("step")
        if s not in id2out:
            return False, f"final.exists references unknown step {s}"
        if id2out[s] != "id_list":
            return False, f"final.exists step {s} must produce id_list"
        return True, ""
    if t == "ratio":
        for key, mode_key in (("numerator_step", "numerator_mode"),
                              ("denominator_step", "denominator_mode")):
            s = final.get(key)
            mode = final.get(mode_key)
            if s not in id2out:
                return False, f"final.ratio references unknown step {s}"
            if mode not in ("count", "scalar"):
                return False, f"final.ratio has invalid {mode_key}: {mode}"
            need = "id_list" if mode == "count" else "scalar"
            if id2out[s] != need:
                return False, f"final.ratio step {s} must produce {need} (mode={mode})"
        return True, ""
    if t == "reduce":
        s = final.get("step")
        if s not in id2out:
            return False, f"final.reduce references unknown step {s}"
        if id2out[s] != "scalar_list":
            return False, f"final.reduce step {s} must produce scalar_list"
        if final.get("op") not in ("sum", "min", "max", "avg", "mean"):
            return False, f"final.reduce has invalid op: {final.get('op')}"
        return True, ""
    return False, f"final.type is missing or invalid: {t}; expected scalar/count/exists/reduce/ratio"


def validate_plan(plan: Dict[str, Any]) -> Tuple[bool, str]:
    steps = plan.get("steps") or []
    if not steps:
        return False, "Plan has no steps"
    ids: List[str] = []
    id2out: Dict[str, str] = {}
    for st in steps:
        sid = st.get("id")
        if not sid:
            return False, "A step is missing its id"
        if not re.fullmatch(_STEP_ID_PATTERN, sid):
            return False, f"Invalid step id: {sid}; it must start with s and contain only letters, digits, and underscores"
        if sid in id2out:
            return False, f"Duplicate step id: {sid}"
        if st.get("output") not in ("id_list", "scalar", "scalar_list"):
            return False, (f"Step {sid} has invalid output: {st.get('output')}"
                           f"; expected id_list, scalar, or scalar_list")
        if not st.get("gremlin"):
            return False, f"Step {sid} has an empty gremlin string"
        ids.append(sid)
        id2out[sid] = st["output"]
    id_set = set(ids)
    dep_map: Dict[str, List[str]] = {}
    for st in steps:
        sid = st["id"]
        for d in (st.get("depends_on") or []):
            if d not in id_set:
                return False, f"Step {sid} depends on unknown step {d}"
            if d == sid:
                return False, f"Step {sid} depends on itself"
        dep_map[sid] = list(st.get("depends_on") or [])
    if _topo_order(dep_map) is None:
        return False, "Step dependencies contain a cycle"
    for st in steps:
        sid = st["id"]
        g = st["gremlin"]
        deps = set(st.get("depends_on") or [])
        for raw in set(_ANY_STEP_PH_RE.findall(g)):
            if not re.fullmatch(_STEP_ID_PATTERN, raw):
                return False, f"Step {sid} uses invalid placeholder {{{{STEP:{raw}}}}}"
            if raw not in id_set:
                return False, (f"Step {sid} references unknown {{{{STEP:{raw}}}}}"
                               f"; defined steps: {'/'.join(ids)}")
        step_refs = set(_STEP_PH_RE.findall(g))
        for r in step_refs:
            if r not in deps:
                return False, f"Step {sid} references {{{{STEP:{r}}}}} but does not declare it in depends_on"
            if id2out.get(r) != "id_list":
                return False, f"Step {sid} references {r}, which does not produce id_list"
        m_empty = re.search(r"\.(or|and|not)\s*\(\s*\)", g)
        if m_empty:
            return False, f"Step {sid} contains empty call .{m_empty.group(1)}()"
        has_seed = bool(_SEED_PH_RE.search(g))
        if not has_seed and not step_refs:
            return False, f"Step {sid} neither anchors {{{{SEED}}}} nor consumes {{{{STEP:sX}}}}"
        if re.search(r"\.V\(\s*\)", g) and not step_refs and not has_seed:
            return False, f"Step {sid} uses an unanchored V() and may scan the full graph"
    ok, err = _validate_final(plan.get("final") or {}, id2out)
    return (True, "") if ok else (False, err)


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
        vals = [float(v) for v in ((scalar_lists.get(step) or [])
                                   if isinstance(step, str) else [])
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


def is_zero_hit_value(value: Any, *, blank_as_zero: bool = False,
                      final_type: str = "", den_zero: bool = False) -> bool:
    if value is None:
        return bool(blank_as_zero)
    if isinstance(value, str) and blank_as_zero and not value.strip():
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if v != 0.0:
        return False
    if str(final_type or "").strip().lower() == "ratio" and not den_zero:
        return False
    return True


def ratio_den_zero_of(trace: Any) -> bool:
    for t in reversed(list(trace or [])):
        note = t.get("note") if isinstance(t, dict) else None
        if isinstance(note, dict) and "ratio_den_zero" in note:
            return True
    return False


class EngineError(Exception):
    pass


Atom = Tuple[Any, ...]


def _unquote(s: str) -> str:
    body = s[1:-1]
    return (body.replace("\\'", "'").replace('\\"', '"')
            .replace("\\\\", "\\").replace("\\n", "\n"))


def _split_top(s: str) -> List[str]:
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


_PRED_NAMES = {"eq", "neq", "gt", "gte", "lt", "lte", "between", "inside", "outside",
               "within", "without", "startswith", "endswith", "containing", "not"}
_PRED_NS = ("P.", "TextP.")


def parse_atom(s: str) -> Atom:
    s = s.strip()
    if not s:
        return ("none", None)
    for ns in _PRED_NS:
        if s.startswith(ns):
            inner = parse_atom(s[len(ns):])
            if inner[0] != "fn":
                raise EngineError(f"Predicate prefix {ns} is not followed by a predicate call: {s[:60]}")
            if inner[1] not in _PRED_NAMES:
                raise EngineError(f"outside the closed predicate set {ns}{inner[1]}(); supported={sorted(_PRED_NAMES)}")
            return inner
    if s[0] in "'\"":
        return ("str", _unquote(s))
    if s.lower() in ("true", "false"):
        return ("bool", s.lower() == "true")
    if re.fullmatch(r"-?\d+", s):
        return ("num", int(s))
    if re.fullmatch(r"-?\d+\.\d+", s):
        return ("num", float(s))
    if s.startswith("__.") or s.startswith("_."):
        return ("trav", parse_chain(s.split(".", 1)[1], anonymous=True))
    if s.startswith("["):
        inner = s[1:-1].strip()
        return ("list", [parse_atom(p) for p in _split_top(inner)] if inner else [])
    m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)", s, re.S)
    if m:
        name = m.group(1)
        args = [parse_atom(p) for p in _split_top(m.group(2))] if m.group(2).strip() else []
        return ("fn", name, args)
    return ("ident", s)


class Step:
    __slots__ = ("name", "args", "raw")

    def __init__(self, name: str, args: List[Atom], raw: str):
        self.name = name
        self.args = args
        self.raw = raw

    def __repr__(self) -> str:
        return f"<{self.name} {self.raw}>"


def parse_chain(text: str, anonymous: bool = False) -> List[Step]:
    s = text.strip()
    if not anonymous:
        m = re.match(r"^(?:__G__|g(?:_[A-Za-z0-9_]+)?)\s*\.", s)
        if not m:
            raise EngineError(f"Chain must start with g.V(...) or __G__.V(...): {text[:120]}")
        s = s[m.end():]
    steps: List[Step] = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] in " .\n\t":
            i += 1
            continue
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", s[i:])
        if not m:
            raise EngineError(f"Cannot parse chain fragment: {s[i:i + 60]}")
        name = m.group(1)
        i += m.end()
        while i < n and s[i] in " \t\n":
            i += 1
        if i >= n or s[i] != "(":
            raise EngineError(f"Operator {name} is missing argument parentheses: {s[max(0, i - 40):i + 20]}")
        depth = 0
        j = i
        in_str = False
        esc = False
        quote = ""
        while j < n:
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
        if depth != 0:
            raise EngineError(f"Operator {name} has unclosed parentheses")
        inner = s[i + 1:j]
        args = [parse_atom(p) for p in _split_top(inner)] if inner.strip() else []
        steps.append(Step(name, args, f"{name}({inner})"))
        i = j + 1
    if not steps:
        raise EngineError(f"Empty traversal: {text[:80]}")
    if not anonymous and steps[0].name != "V":
        raise EngineError(f"Chain must start with V(...) anchored by hasId: {text[:120]}")
    return steps


def _as_num(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (np.integer, np.floating)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


# Numeric literals in the vertex-property domain intentionally cannot match strings.
_NO_MATCH = object()


def _cmp_eq(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    if a is _NO_MATCH or b is _NO_MATCH:
        return False
    na, nb = _as_num(a), _as_num(b)
    if na is not None and nb is not None:
        return na == nb
    return str(a) == str(b)


def _cmp_order(a: Any, b: Any, op: str) -> bool:
    if a is None or b is None:
        return False
    if a is _NO_MATCH or b is _NO_MATCH:
        return False
    na, nb = _as_num(a), _as_num(b)
    if na is not None and nb is not None:
        pass
    else:
        na, nb = str(a), str(b)
    if op == "gt":
        return na > nb
    if op == "gte":
        return na >= nb
    if op == "lt":
        return na < nb
    if op == "lte":
        return na <= nb
    return False


def _flatten_atoms(atoms: Sequence[Atom]) -> List[Atom]:
    out: List[Atom] = []
    for a in atoms:
        if a[0] == "list":
            out.extend(_flatten_atoms(a[1]))
        else:
            out.append(a)
    return out


def _match_pred(value: Any, atom: Atom) -> bool:
    kind = atom[0]
    if kind == "unmatchable":
        return False
    if kind == "fn":
        name = atom[1]
        args = _flatten_atoms(atom[2])
        if name != "not" and any(a[0] == "fn" for a in args):
            raise EngineError(
                f"Nested predicate {name}(<predicate>) has ambiguous semantics; server behavior is not verified. "
                f"For example, neq(within(...)) is always true in TinkerPop and does not filter; rewrite using within/without")
        arity = {"eq": 1, "neq": 1, "gt": 1, "gte": 1, "lt": 1, "lte": 1,
                 "between": 2, "inside": 2, "outside": 2,
                 "startswith": 1, "endswith": 1, "containing": 1}
        if name in arity and len(args) != arity[name]:
            raise EngineError(f"Predicate {name}() requires {arity[name]} arguments, got {len(args)}")
        if name == "eq":
            return _cmp_eq(value, _atom_value(args[0]))
        if name == "neq":
            return not _cmp_eq(value, _atom_value(args[0]))
        if name in ("gt", "gte", "lt", "lte"):
            return _cmp_order(value, _atom_value(args[0]), name)
        if name == "within":
            return any(_cmp_eq(value, _atom_value(a)) for a in args)
        if name == "without":
            return not any(_cmp_eq(value, _atom_value(a)) for a in args)
        if name == "between":
            lo = _atom_value(args[0]) if args else None
            hi = _atom_value(args[1]) if len(args) > 1 else None
            return _cmp_order(value, lo, "gte") and _cmp_order(value, hi, "lt")
        if name in ("inside", "outside"):
            lo = _atom_value(args[0]) if args else None
            hi = _atom_value(args[1]) if len(args) > 1 else None
            inside = _cmp_order(value, lo, "gt") and _cmp_order(value, hi, "lt")
            return inside if name == "inside" else not inside
        if name in ("startswith", "endswith", "containing"):
            s, needle = str(value or ""), str(_atom_value(args[0]) if args else "")
            if name == "startswith":
                return s.startswith(needle)
            if name == "endswith":
                return s.endswith(needle)
            return needle in s
        if name == "not":
            return not any(_match_pred(value, a) for a in args)
        raise EngineError(f"outside the closed predicate set {name}(); supported={sorted(_PRED_NAMES)}")
    return _cmp_eq(value, _atom_value(atom))


def _atom_value(atom: Atom) -> Any:
    kind = atom[0]
    if kind in ("str", "num", "bool"):
        return atom[1]
    if kind == "none":
        return None
    if kind == "unmatchable":
        return _NO_MATCH
    if kind == "ident":
        key = str(atom[1])
        return {"true": True, "false": False}.get(key.lower(), key)
    if kind == "list":
        return [_atom_value(a) for a in atom[1]]
    raise EngineError(f"Cannot evaluate atom {atom}")


def _atom_str_domain(atom: Atom) -> Atom:
    """Map numeric vertex-property literals to a sentinel that never matches strings.

    ``hasNot`` and ``neq`` continue to use type-aware inequality semantics.
    """
    kind = atom[0]
    if kind == "num":
        return ("unmatchable", None)
    if kind == "fn":
        return ("fn", atom[1], [_atom_str_domain(a) for a in atom[2]])
    if kind == "list":
        return ("list", [_atom_str_domain(a) for a in atom[1]])
    return atom


SUPPORTED = {
    "V", "hasId", "has", "hasNot", "outE", "inE", "bothE", "outV", "inV", "otherV",
    "bothV", "out", "in", "both", "dedup", "id", "values", "count", "sum", "mean",
    "min", "max", "toList", "limit", "range", "order", "by", "where", "not",
    "filter", "is", "and", "or", "union", "local", "fold", "unfold", "constant",
    "identity",
}
_NUMERIC = {"total_amount", "trade_count", "top_k"}

_AMOUNT_PROPS = frozenset({"total_amount"})
_AMOUNT_FEN2YUAN = 100.0
_AMOUNT_AGG_RE = re.compile(
    r"values\s*\(\s*['\"](total_amount)['\"]\s*\)\s*\.\s*(sum|mean|avg|min|max|fold)\s*\(",
    re.IGNORECASE)


class Traversal:
    __slots__ = ("kind", "data", "anchor", "value_prop", "order_flag")

    def __init__(self, kind: str, data: Any, anchor: Any = None,
                 value_prop: Optional[str] = None):
        self.kind = kind
        self.data = data
        self.anchor = anchor
        self.value_prop = value_prop
        self.order_flag = False

    def __len__(self) -> int:
        if self.kind in ("V", "E", "values"):
            return len(self.data)
        return 0


class LocalExecutor:
    def __init__(self, store: GraphStore, *, amount_unit: str = "yuan"):
        self.store = store
        self.amount_unit = amount_unit
        self.type_warnings: List[str] = []

    def run_chain(self, steps: List[Step]) -> Traversal:
        head = steps[0]
        i = 1
        if not head.args and len(steps) > 1 and steps[1].name == "hasId":
            ids: List[str] = []
            for a in steps[1].args:
                ids.extend(self._ids_of(a))
            known = [u for u in dict.fromkeys(ids) if self.store.has_vertex(u)]
            state = Traversal("V", np.array(known, dtype=object))
            i = 2
        else:
            state = self._do_V(head)
        while i < len(steps):
            st = steps[i]
            name = st.name
            if name == "by":
                state = self._do_by(state, st)
                i += 1
                continue
            if name not in SUPPORTED:
                raise EngineError(
                    f"outside the closed operator set {name}(); this engine supports only the FraudFeat operator set. "
                    f"The feature is invalid")
            handler = getattr(self, f"_do_{name}", None)
            if handler is None:
                raise EngineError(f"Operator {name}() is not implemented; CLOSED_SET and implementation disagree")
            state = handler(state, st)
            i += 1
        return state

    def _ids_of(self, atom: Atom) -> List[str]:
        if atom[0] == "list":
            out: List[str] = []
            for a in atom[1]:
                out.extend(self._ids_of(a))
            return out
        if atom[0] == "fn" and atom[1] == "within":
            out = []
            for a in atom[2]:
                out.extend(self._ids_of(a))
            return out
        v = _atom_value(atom)
        return [str(v)] if v not in (None, "") else []

    def _do_V(self, st: Step) -> Traversal:
        if not st.args:
            ids = np.array([u for u, _e in self.store.iter_vertices()], dtype=object)
            return Traversal("V", ids)
        ids: List[str] = []
        for a in st.args:
            ids.extend(self._ids_of(a))
        known = [u for u in dict.fromkeys(ids) if self.store.has_vertex(u)]
        return Traversal("V", np.array(known, dtype=object))

    def _do_hasId(self, state: Traversal, st: Step) -> Traversal:
        ids: List[str] = []
        for a in st.args:
            ids.extend(self._ids_of(a))
        wanted = set(ids)
        if state.kind == "V":
            data = np.array([u for u in state.data if str(u) in wanted], dtype=object)
            return Traversal("V", data)
        if state.kind == "E":
            eids = self.store.edge_values(state.data, "edge_id")
            keep = np.array([str(x) in wanted for x in eids], dtype=bool)
            return Traversal("E", state.data[keep], state.anchor)
        raise EngineError("hasId() can only be applied to vertices or edges")

    def _prop_values(self, state: Traversal, prop: str) -> np.ndarray:
        if state.kind == "V":
            return np.array([self.store.node_value(str(u), prop) for u in state.data],
                            dtype=object)
        if state.kind == "E":
            return self.store.edge_values(state.data, prop)
        raise EngineError(f"has('{prop}', ...) can only be applied to vertices or edges")

    def _scale_amount(self, prop: str, atom: Atom) -> Atom:
        if self.amount_unit != "yuan" or prop != "total_amount":
            return atom
        if atom[0] == "num":
            return ("num", float(atom[1]) * 100.0)
        if atom[0] == "fn":
            return ("fn", atom[1], [self._scale_amount(prop, a) for a in atom[2]])
        if atom[0] == "list":
            return ("list", [self._scale_amount(prop, a) for a in atom[1]])
        return atom

    def _numeric_prop(self, state: Traversal, prop: str) -> bool:
        if state.kind == "E":
            return prop in _NUMERIC
        if state.kind == "values":
            return prop in _NUMERIC
        return False

    @staticmethod
    def _prop_present(v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, float) and math.isnan(v):
            return False
        return str(v) != ""

    def _do_has(self, state: Traversal, st: Step) -> Traversal:
        if len(st.args) == 1:
            # has('prop') tests property presence.
            prop = str(_atom_value(st.args[0]))
            vals = self._prop_values(state, prop)
            keep = np.array([self._prop_present(v) for v in vals], dtype=bool)
            return Traversal(state.kind, state.data[keep],
                             None if state.anchor is None else state.anchor[keep])
        if len(st.args) == 3:
            # has(label, prop, value): the first argument is the label.
            prop_atom, val_atom = st.args[1], st.args[2]
        elif len(st.args) == 2:
            prop_atom, val_atom = st.args[0], st.args[1]
        else:
            raise EngineError(f"Invalid has() argument count: {st.raw}")
        prop = str(_atom_value(prop_atom))
        if self._numeric_prop(state, prop):
            val_atom = self._scale_amount(prop, val_atom)
        else:
            if any(a[0] == "num" for a in _flatten_atoms([val_atom])):
                self.type_warnings.append(
                    f"has('{prop}', <numeric literal>) cannot match a string "
                    "vertex property; quote the value explicitly")
            val_atom = _atom_str_domain(val_atom)
        vals = self._prop_values(state, prop)
        keep = np.array([_match_pred(v, val_atom) for v in vals], dtype=bool)
        return Traversal(state.kind, state.data[keep],
                         None if state.anchor is None else state.anchor[keep])

    def _do_hasNot(self, state: Traversal, st: Step) -> Traversal:
        if len(st.args) != 1:
            raise EngineError(f"hasNot() requires one property-name argument: {st.raw}")
        prop = str(_atom_value(st.args[0]))
        vals = self._prop_values(state, prop)
        keep = np.array([not self._prop_present(v) for v in vals], dtype=bool)
        return Traversal(state.kind, state.data[keep],
                         None if state.anchor is None else state.anchor[keep])

    def _edges_from_vertices(self, state: Traversal, direction: str) -> Traversal:
        if state.kind != "V":
            raise EngineError(f"{direction}E() can only be applied to vertices")
        parts: List[np.ndarray] = []
        anchors: List[np.ndarray] = []
        for u in state.data:
            pos = self.store.incident(str(u), direction.replace("E", ""))
            parts.append(pos)
            anchors.append(np.array([str(u)] * len(pos), dtype=object))
        pos = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        anc = np.concatenate(anchors) if anchors else np.empty(0, dtype=object)
        return Traversal("E", pos, anc)

    def _do_outE(self, state: Traversal, st: Step) -> Traversal:
        return self._filter_edge_labels(self._edges_from_vertices(state, "out"), st)

    def _do_inE(self, state: Traversal, st: Step) -> Traversal:
        return self._filter_edge_labels(self._edges_from_vertices(state, "in"), st)

    def _do_bothE(self, state: Traversal, st: Step) -> Traversal:
        return self._filter_edge_labels(self._edges_from_vertices(state, "both"), st)

    def _filter_edge_labels(self, state: Traversal, st: Step) -> Traversal:
        if not st.args:
            return state
        labels: List[str] = []
        for a in st.args:
            v = _atom_value(a)
            labels.extend(v if isinstance(v, list) else [v])
        rels = self.store.edge_values(state.data, "relation")
        wanted = {str(label) for label in labels}
        keep = np.array([str(r) in wanted for r in rels], dtype=bool)
        return Traversal("E", state.data[keep], state.anchor[keep])

    def _vertices_from_edges(self, state: Traversal, side: str) -> Traversal:
        if state.kind != "E":
            raise EngineError(f"{side}V() can only be applied to edges")
        src = self.store.edge_endpoint(state.data, "src")
        if side == "out":
            return Traversal("V", np.array(src, dtype=object))
        dst = self.store.edge_endpoint(state.data, "dst")
        if side == "in":
            return Traversal("V", np.array(dst, dtype=object))
        if side == "other":
            if state.anchor is None:
                raise EngineError("otherV() requires a preceding vertex anchor; the chain context is missing")
            out = np.array([d if str(s) == str(a) else s
                            for s, d, a in zip(src, dst, state.anchor)], dtype=object)
            return Traversal("V", out)
        raise EngineError(f"Unknown vertex direction {side}")

    def _do_outV(self, state: Traversal, st: Step) -> Traversal:
        return self._vertices_from_edges(state, "out")

    def _do_inV(self, state: Traversal, st: Step) -> Traversal:
        return self._vertices_from_edges(state, "in")

    def _do_otherV(self, state: Traversal, st: Step) -> Traversal:
        return self._vertices_from_edges(state, "other")

    def _do_bothV(self, state: Traversal, st: Step) -> Traversal:
        a = self._vertices_from_edges(state, "out")
        b = self._vertices_from_edges(state, "in")
        return Traversal("V", np.concatenate([a.data, b.data]))

    def _do_out(self, state: Traversal, st: Step) -> Traversal:
        e = self._filter_edge_labels(self._edges_from_vertices(state, "out"), st)
        return self._vertices_from_edges(e, "in")

    def _do_in(self, state: Traversal, st: Step) -> Traversal:
        e = self._filter_edge_labels(self._edges_from_vertices(state, "in"), st)
        return self._vertices_from_edges(e, "out")

    def _do_both(self, state: Traversal, st: Step) -> Traversal:
        e = self._filter_edge_labels(self._edges_from_vertices(state, "both"), st)
        return self._vertices_from_edges(e, "other")

    def _do_dedup(self, state: Traversal, st: Step) -> Traversal:
        if state.kind not in ("V", "E", "values"):
            return state
        if state.kind == "E":
            _u, idx = np.unique(state.data, return_index=True)
            order = np.sort(idx)
            return Traversal("E", state.data[order], state.anchor[order])
        seen: set = set()
        out = []
        for x in state.data:
            key = str(x)
            if key in seen:
                continue
            seen.add(key)
            out.append(x)
        return Traversal(state.kind, np.array(out, dtype=object),
                         value_prop=state.value_prop)

    def _do_id(self, state: Traversal, st: Step) -> Traversal:
        if state.kind == "V":
            return Traversal("values", np.array([str(u) for u in state.data], dtype=object))
        if state.kind == "E":
            return Traversal("values", self.store.edge_values(state.data, "edge_id"))
        raise EngineError("id() can only be applied to vertices or edges")

    def _do_values(self, state: Traversal, st: Step) -> Traversal:
        if not st.args:
            raise EngineError("values() requires a property name")
        props = [str(_atom_value(a)) for a in st.args]
        vals = self._prop_values(state, props[0])
        if len(props) > 1:
            alt = self._prop_values(state, props[1])
            vals = np.array([a if a not in (None, "") else b for a, b in zip(vals, alt)],
                            dtype=object)
        return Traversal("values", vals, value_prop=props[0])

    def _values_numeric(self, state: Traversal) -> np.ndarray:
        if state.kind != "values":
            raise EngineError("Aggregation operators can only be applied to values(...)")
        nums = [_as_num(v) for v in state.data]
        return np.array([n for n in nums if n is not None], dtype="float64")

    def _do_count(self, state: Traversal, st: Step) -> Traversal:
        if st.args:
            arg = _atom_value(st.args[0])
            if not (isinstance(arg, str) and arg.lower() == "local"):
                raise EngineError(f"Invalid count() argument: {st.raw}; expected count() or count(local)")
        return Traversal("scalar", len(state))

    def _do_sum(self, state: Traversal, st: Step) -> Traversal:
        arr = self._values_numeric(state)
        return Traversal("scalar", float(arr.sum()) if len(arr) else 0.0)

    def _do_mean(self, state: Traversal, st: Step) -> Traversal:
        arr = self._values_numeric(state)
        return Traversal("scalar", float(arr.mean()) if len(arr) else 0.0)

    def _do_min(self, state: Traversal, st: Step) -> Traversal:
        arr = self._values_numeric(state)
        return Traversal("scalar", float(arr.min()) if len(arr) else 0.0)

    def _do_max(self, state: Traversal, st: Step) -> Traversal:
        arr = self._values_numeric(state)
        return Traversal("scalar", float(arr.max()) if len(arr) else 0.0)

    def _do_toList(self, state: Traversal, st: Step) -> Traversal:
        return state

    def _do_limit(self, state: Traversal, st: Step) -> Traversal:
        n = int(_atom_value(st.args[0])) if st.args else 0
        if state.kind == "E":
            return Traversal("E", state.data[:n], state.anchor[:n])
        if state.kind in ("V", "values"):
            return Traversal(state.kind, state.data[:n], value_prop=state.value_prop)
        return state

    def _do_range(self, state: Traversal, st: Step) -> Traversal:
        lo = int(_atom_value(st.args[0])) if st.args else 0
        hi = int(_atom_value(st.args[1])) if len(st.args) > 1 else len(state)
        if state.kind == "E":
            return Traversal("E", state.data[lo:hi], state.anchor[lo:hi])
        if state.kind in ("V", "values"):
            return Traversal(state.kind, state.data[lo:hi], value_prop=state.value_prop)
        return state

    def _do_order(self, state: Traversal, st: Step) -> Traversal:
        state.order_flag = True
        return state

    def _do_by(self, state: Traversal, st: Step) -> Traversal:
        if not state.order_flag:
            raise EngineError("by() is supported only after order(); grouping/projection by() is outside the closed set")
        prop: Optional[str] = None
        desc = False
        for a in st.args:
            if a[0] == "str":
                prop = a[1]
            elif a[0] == "ident" and str(a[1]).lower() in ("desc", "incr", "asc"):
                desc = str(a[1]).lower() == "desc"
            elif a[0] == "fn" and a[1] == "values" and a[2]:
                prop = str(_atom_value(a[2][0]))
            elif a[0] == "trav" and a[1] and a[1][0].name == "values":
                prop = str(_atom_value(a[1][0].args[0]))
        if prop is None:
            raise EngineError(f"order().by() requires a property name: {st.raw}")
        if state.kind not in ("V", "E"):
            raise EngineError("order().by() supports only vertex or edge sets")
        vals = self._prop_values(state, prop)
        if self._numeric_prop(state, prop):
            nums = np.array([np.nan if _as_num(v) is None else float(_as_num(v))
                             for v in vals])
            idx = np.argsort(np.where(np.isnan(nums), np.inf, nums))
        else:
            idx = np.argsort(np.array([str(v) for v in vals], dtype=object))
        if desc:
            idx = idx[::-1]
        if state.kind == "E":
            return Traversal("E", state.data[idx], state.anchor[idx])
        return Traversal("V", state.data[idx])

    def _sub_filter(self, state: Traversal, trav: List[Step], mode: str) -> Traversal:
        keep: List[bool] = []
        for i in range(len(state.data)):
            started = self._start_at(state, i)
            try:
                out = self.run_sub(trav, started)
                truth = self._truthy(out)
            except EngineError:
                raise
            keep.append(truth if mode == "where" else (not truth))
        mask = np.array(keep, dtype=bool)
        if state.kind == "E":
            return Traversal("E", state.data[mask], state.anchor[mask])
        return Traversal(state.kind, state.data[mask], value_prop=state.value_prop)

    def _start_at(self, state: Traversal, i: int) -> Traversal:
        if state.kind == "V":
            return Traversal("V", np.array([state.data[i]], dtype=object))
        if state.kind == "E":
            anc = None if state.anchor is None else np.array([state.anchor[i]], dtype=object)
            return Traversal("E", np.array([state.data[i]], dtype=np.int64), anc)
        if state.kind == "values":
            return Traversal("values", np.array([state.data[i]], dtype=object),
                             value_prop=state.value_prop)
        return Traversal(state.kind, state.data)

    @staticmethod
    def _truthy(out: Traversal) -> bool:
        if out.kind in ("V", "E", "values"):
            return len(out.data) > 0
        if out.kind == "scalar":
            v = out.data
            return bool(v) if not isinstance(v, str) else bool(v.strip())
        return bool(out.data)

    def run_sub(self, steps: List[Step], start: Traversal) -> Traversal:
        state = start
        i = 0
        while i < len(steps):
            st = steps[i]
            if st.name == "by":
                state = self._do_by(state, st)
                i += 1
                continue
            if st.name not in SUPPORTED:
                raise EngineError(f"outside the closed operator set {st.name}() inside a sub-traversal")
            handler = getattr(self, f"_do_{st.name}", None)
            if handler is None:
                raise EngineError(f"Operator {st.name}() is not implemented")
            state = handler(state, st)
            i += 1
        return state

    def _do_where(self, state: Traversal, st: Step) -> Traversal:
        trav = self._first_trav(st)
        return self._sub_filter(state, trav, "where")

    def _do_filter(self, state: Traversal, st: Step) -> Traversal:
        trav = self._first_trav(st)
        return self._sub_filter(state, trav, "where")

    def _do_not(self, state: Traversal, st: Step) -> Traversal:
        trav = self._first_trav(st)
        return self._sub_filter(state, trav, "not")

    @staticmethod
    def _first_trav(st: Step) -> List[Step]:
        for a in st.args:
            if a[0] == "trav":
                return a[1]
        raise EngineError(f"{st.name}() requires an anonymous sub-traversal (__.xxx): {st.raw}")

    def _do_and(self, state: Traversal, st: Step) -> Traversal:
        travs = [a[1] for a in st.args if a[0] == "trav"]
        if not travs:
            raise EngineError("and() requires an anonymous sub-traversal")
        for trav in travs:
            state = self._sub_filter(state, trav, "where")
        return state

    def _do_or(self, state: Traversal, st: Step) -> Traversal:
        travs = [a[1] for a in st.args if a[0] == "trav"]
        if not travs:
            raise EngineError("or() requires an anonymous sub-traversal")
        keep: List[bool] = []
        for i in range(len(state.data)):
            started = self._start_at(state, i)
            ok = False
            for trav in travs:
                if self._truthy(self.run_sub(trav, started)):
                    ok = True
                    break
            keep.append(ok)
        mask = np.array(keep, dtype=bool)
        if state.kind == "E":
            return Traversal("E", state.data[mask], state.anchor[mask])
        return Traversal(state.kind, state.data[mask], value_prop=state.value_prop)

    def _do_is(self, state: Traversal, st: Step) -> Traversal:
        if not st.args:
            raise EngineError("is() requires a predicate")
        pred = st.args[0]
        if state.kind == "scalar":
            return state if _match_pred(state.data, pred) else Traversal("values",
                                                                        np.array([], dtype=object))
        if state.kind == "values":
            if not self._numeric_prop(state, state.value_prop or ""):
                pred = _atom_str_domain(pred)
            keep = np.array([_match_pred(v, pred) for v in state.data], dtype=bool)
            return Traversal("values", state.data[keep], value_prop=state.value_prop)
        if state.kind == "V":
            keep = np.array([_match_pred(str(u), pred) for u in state.data], dtype=bool)
            return Traversal("V", state.data[keep])
        if state.kind == "E":
            raise EngineError("is() cannot be applied directly to edge sets; call .id() before is(...)")
        raise EngineError("is() supports only scalars, values, and vertex sets")

    def _do_union(self, state: Traversal, st: Step) -> Traversal:
        travs = [a[1] for a in st.args if a[0] == "trav"]
        if not travs:
            raise EngineError("union() requires anonymous sub-traversals")
        kind = None
        chunks: List[Any] = []
        for i in range(len(state.data)):
            started = self._start_at(state, i)
            for trav in travs:
                out = self.run_sub(trav, started)
                kind = kind or out.kind
                if out.kind != kind:
                    raise EngineError("union() branches return inconsistent types")
                chunks.append(out.data)
        if kind is None:
            return Traversal("values", np.array([], dtype=object))
        if kind == "E":
            return Traversal("E", np.concatenate(chunks) if chunks else
                             np.empty(0, dtype=np.int64))
        return Traversal(kind, np.concatenate(chunks) if chunks else
                         np.array([], dtype=object))

    def _do_local(self, state: Traversal, st: Step) -> Traversal:
        trav = self._first_trav(st)
        kind = None
        chunks: List[Any] = []
        for i in range(len(state.data)):
            out = self.run_sub(trav, self._start_at(state, i))
            kind = kind or out.kind
            chunks.append(out.data)
        if kind is None:
            return Traversal("values", np.array([], dtype=object))
        return Traversal(kind, np.concatenate(chunks) if chunks else
                         np.array([], dtype=object))

    def _do_fold(self, state: Traversal, st: Step) -> Traversal:
        return Traversal("values", np.array([list(state.data)], dtype=object))

    def _do_unfold(self, state: Traversal, st: Step) -> Traversal:
        if state.kind != "values":
            raise EngineError("unfold() can only be applied to values")
        out: List[Any] = []
        for v in state.data:
            if isinstance(v, (list, tuple, np.ndarray)):
                out.extend(list(v))
            else:
                out.append(v)
        return Traversal("values", np.array(out, dtype=object), value_prop=state.value_prop)

    def _do_constant(self, state: Traversal, st: Step) -> Traversal:
        val = _atom_value(st.args[0]) if st.args else None
        if state.kind in ("V", "E"):
            n = len(state.data)
            return Traversal("values", np.array([val] * n, dtype=object))
        return Traversal("values", np.array([val], dtype=object))

    def _do_identity(self, state: Traversal, st: Step) -> Traversal:
        return state

    def _do_as(self, state: Traversal, st: Step) -> Traversal:
        return state

    def _do_select(self, state: Traversal, st: Step) -> Traversal:
        return state


def _resolve_gremlin(gremlin: str, seed_uid: str, inter: Dict[str, List[Any]]) -> str:
    g = _QUOTED_SEED_PH_RE.sub(lambda m: m.group(1), gremlin)
    g = g.replace("{{SEED}}", json.dumps(str(seed_uid), ensure_ascii=False))
    g = _QUOTED_STEP_PH_RE.sub(lambda m: m.group(1), g)

    def _repl(m: "re.Match[str]") -> str:
        sid = m.group(1)
        vals = [str(x) for x in (inter.get(sid) or [])]
        return json.dumps(vals, ensure_ascii=False)

    return _STEP_PH_RE.sub(_repl, g)


def _execute_plan_local_impl(store: GraphStore, plan: Dict[str, Any], seed_unique_id: str,
                             *, amount_unit: str = "fen",
                             id_list_limit: int = 300) -> Tuple[Any, List[Dict[str, Any]], str]:
    plan = normalize_plan(plan)
    ok, err = validate_plan(plan)
    if not ok:
        return None, [], f"Invalid plan structure: {err}"
    steps = plan["steps"]
    dep_map = {st["id"]: list(st.get("depends_on") or []) for st in steps}
    levels = _topo_order(dep_map)
    if levels is None:
        return None, [], "Dependencies contain a cycle"
    inter: Dict[str, List[Any]] = {}
    scalars: Dict[str, Any] = {}
    scalar_lists: Dict[str, List[Any]] = {}
    trace: List[Dict[str, Any]] = []
    ex = LocalExecutor(store, amount_unit=amount_unit)
    by_id = {st["id"]: st for st in steps}
    for level in levels:
        for sid in level:
            st = by_id[sid]
            resolved = _resolve_gremlin(st["gremlin"], seed_unique_id, inter)
            is_amt_agg = bool(_AMOUNT_AGG_RE.search(resolved))
            rec: Dict[str, Any] = {"id": sid, "desc": st.get("desc"),
                                   "output": st.get("output"),
                                   "resolved_gremlin": resolved}
            if is_amt_agg and amount_unit == "yuan":
                rec["amt_unit"] = "total_amount aggregates are converted fen to yuan (/100); filters are converted yuan to fen (x100)"
            warn_before = len(ex.type_warnings)
            try:
                parsed = parse_chain(resolved)
                state = ex.run_chain(parsed)
                if len(ex.type_warnings) > warn_before:
                    rec["warnings"] = ex.type_warnings[warn_before:]
                out_type = st.get("output")
                if out_type == "id_list":
                    if state.kind == "V":
                        ids = [str(u) for u in state.data]
                    elif state.kind == "values":
                        ids = [str(v) for v in state.data if v not in (None, "")]
                    else:
                        raise EngineError(f"Step {sid} declares id_list but the chain does not end in a vertex/ID set")
                    ids = list(dict.fromkeys(ids))
                    if id_list_limit and id_list_limit > 0:
                        ids = ids[:id_list_limit]
                    inter[sid] = ids
                    rec["result_preview"] = f"{len(ids)} IDs"
                elif out_type == "scalar":
                    if state.kind != "scalar":
                        raise EngineError(f"Step {sid} declares scalar but the chain does not end in a scalar")
                    val = state.data
                    if is_amt_agg and amount_unit == "yuan" and isinstance(val, (int, float)):
                        val = round(float(val) / _AMOUNT_FEN2YUAN, 2)
                    scalars[sid] = val
                    rec["result_preview"] = (f"scalar={val}"
                                             + (" (yuan)" if is_amt_agg and amount_unit == "yuan" else ""))
                elif out_type == "scalar_list":
                    if state.kind not in ("values", "scalar"):
                        raise EngineError(f"Step {sid} declares scalar_list but the chain does not end in a numeric collection")
                    vals = ([state.data] if state.kind == "scalar"
                            else list(state.data))
                    flat: List[Any] = []
                    for v in vals:
                        if isinstance(v, (list, tuple, np.ndarray)):
                            flat.extend(list(v))
                        else:
                            flat.append(v)
                    nums = [v for v in flat if _as_num(v) is not None]
                    if is_amt_agg and amount_unit == "yuan":
                        nums = [round(float(v) / _AMOUNT_FEN2YUAN, 2) for v in nums]
                    scalar_lists[sid] = nums
                    rec["result_preview"] = (f"{len(nums)} numeric values"
                                             + (" (yuan)" if is_amt_agg and amount_unit == "yuan" else ""))
                else:
                    raise EngineError(f"Step {sid} has invalid output: {out_type}")
            except EngineError as e:
                rec["error"] = str(e)
                trace.append(rec)
                return None, trace, f"Step {sid} failed: {e}"
            trace.append(rec)
    note: Dict[str, Any] = {}
    value = _final_reduce(plan.get("final") or {}, inter, scalars, scalar_lists, note)
    if note:
        trace.append({"id": "__final__", "note": note})
    return value, trace, ""


def execute_plan_local(store: GraphStore, plan: Dict[str, Any], seed_unique_id: str,
                       *, amount_unit: str = "fen",
                       id_list_limit: int = 300) -> Tuple[Any, List[Dict[str, Any]], str]:
    """Execute one feature without allowing malformed input to abort the whole batch."""
    try:
        return _execute_plan_local_impl(
            store, plan, seed_unique_id, amount_unit=amount_unit,
            id_list_limit=id_list_limit)
    except Exception as exc:  # noqa: BLE001
        return None, [], f"Unexpected local execution error: {type(exc).__name__}: {exc}"
