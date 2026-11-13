# -*- coding: utf-8 -*-
"""Structural diversity from anchored graph edit distance and statistic signatures.

Each plan becomes a directed entity-type multigraph. Nodes represent traversal
positions, parallel relations remain parallel edges, branches are preserved, and
property filters affect only the statistic signature. ``outE``, ``inE``, and
``bothE`` are supported; relation endpoints from the public schema constrain which
edges can be attached to each typed position.

For features f and g::

    S(f,g) = 0.75 * (1 - GED(f,g) / max(1, Omega_f + Omega_g))
             + 0.25 * S_stat(f,g)

The statistic signature weights operator, measured object, deduplication path, and
expected value type by 0.3/0.3/0.2/0.2. Set metrics include strict structure count,
mean pairwise similarity, duplicate ratio, near-duplicate ratios at tau=0.70..0.95,
and connected redundancy clusters.

The analysis pool is the input prefix selected before IV ranking. Entries with the
same feature name remain distinct by pool index.

Example::

    python -m fraudfeat.diversity --features features.jsonl \\
        --schema data/schema.json --out-dir results --pool-limit 100
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from itertools import permutations, product
from math import factorial
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

from .pool import analysis_pool, entry_key, load_features, load_iv_jsonl, prepare_features, ranked_indices
from .structure_pattern import plan_from_graph_pattern


__all__ = ["SchemaIndex", "build_graph", "graph_edit_distance", "graph_similarity",
           "stat_signature", "signature_key", "structure_similarity",
           "batch_metrics", "summarize", "main"]
logger = logging.getLogger("fraudfeat.diversity")
TAU = 0.95
TAU_SWEEP = tuple(round(0.70 + 0.05 * i, 2) for i in range(6))
W_GRAPH, W_STAT = 0.75, 0.25
W_OP, W_MU, W_DEDUP, W_EVT = 0.30, 0.30, 0.20, 0.20
_UNKNOWN = "?"
_GED_VISIT_CAP = 200_000
_CANON_PERM_CAP = 60_000

_STEP_RE = re.compile(r"\{\{STEP:([A-Za-z0-9_]+)\}\}")
_SEED_RE = re.compile(r"\{\{SEED\}\}")
_EVENT_RE = re.compile(
    r"\.(?P<edge>outE|inE|bothE)\s*\((?P<elabel>[^)]*)\)"
    r"|has\(\s*['\"]relation['\"]\s*,\s*(?:within\((?P<relation_list>[^)]*)\)"
    r"|['\"](?P<relation_name>[^'\"]+)['\"])"
    r"|\.(?P<varr>inV|outV|otherV)\s*\(\s*\)"
    r"|\.values\(\s*['\"](?P<val>[^'\"]+)['\"]\s*\)"
    r"|\.(?P<agg>sum|mean|min|max|count)\s*\(\s*\)"
    r"|\.dedup\s*\(\s*\)")
_QUOTED_RE = re.compile(r"['\"]([^'\"]+)['\"]")


class _TG(NamedTuple):
    types: Tuple[str, ...]
    edges: Tuple[Tuple[int, int, str, str], ...]


class _StepInfo(NamedTuple):
    measures: tuple
    dedups: tuple
    agg: str
    end_on_edge: bool
    last_edges: tuple


class SchemaIndex:
    """Index public relation names and their source/destination entity labels."""

    def __init__(self, schema: Dict[str, Any]):
        self.short: Dict[str, str] = {}
        entities = list(schema.get("entities") or [])
        for entity in entities:
            entity_type = str(entity.get("entity_type") or "")
            if entity_type:
                self.short[entity_type] = str(entity.get("short") or entity_type)
        self.public_rel: Dict[str, Tuple[str, str, str]] = {}
        for relation in schema.get("relations") or []:
            public_name = str(relation.get("en") or "")
            for endpoint in relation.get("join_fields") or []:
                fs = self.short.get(str(endpoint.get("src")), str(endpoint.get("src")))
                ts = self.short.get(str(endpoint.get("dst")), str(endpoint.get("dst")))
                if public_name:
                    self.public_rel[public_name] = (public_name, fs, ts)

    @classmethod
    def load(cls, path: str | Path) -> "SchemaIndex":
        p = Path(path)
        data_dir = p.parent if (p.is_file() or p.suffix.lower() == ".json") else p
        from .graph_store import load_schema
        return cls(load_schema(data_dir))

    def rel_meta(self, relation_key: str) -> Tuple[str, str, str]:
        return self.public_rel.get(
            relation_key, (relation_key or _UNKNOWN, _UNKNOWN, _UNKNOWN))


def _type_compat(position_type: str, required_type: str) -> bool:
    """Allow an edge only when its source-side endpoint matches the position type."""
    if (not position_type or not required_type or position_type == _UNKNOWN
            or required_type == _UNKNOWN):
        return True
    return position_type == required_type


def _plan_of(f: Any) -> Optional[Dict[str, Any]]:
    plan = f.get("gremlin_plan") if isinstance(f, dict) else None
    if plan is None and isinstance(f, dict):
        plan = f.get("plan")
    if isinstance(plan, dict) and plan.get("steps"):
        return plan
    if isinstance(f, dict) and f.get("graph_pattern") is not None:
        try:
            return plan_from_graph_pattern(f.get("graph_pattern"))
        except (TypeError, ValueError):
            return None
    return None


def _path_str(path: tuple) -> str:
    return ">".join(f"{r}:{d}" for r, d in path)


def _topo(steps: List[dict], by_id: Dict[str, dict]) -> List[str]:
    ids = [str(s.get("id")) for s in steps if s.get("id")]
    deps = {sid: [r for r in _STEP_RE.findall(str(by_id[sid].get("gremlin") or ""))
                  if r in by_id] for sid in ids}
    done, out = set(), []
    remaining = list(ids)
    while remaining:
        progressed = False
        for sid in list(remaining):
            if all(d in done for d in deps[sid]):
                out.append(sid)
                done.add(sid)
                remaining.remove(sid)
                progressed = True
        if not progressed:
            out.extend(remaining)
            break
    return out


def _extract(f: Any, idx: SchemaIndex):
    plan = _plan_of(f)
    types: List[str] = [""]
    edges: List[Tuple[int, int, str, str]] = []
    path_of: Dict[int, tuple] = {0: ()}
    if plan is None:
        return _TG(tuple(types), ()), (("",), (), (), _kappa(f))
    steps = [s for s in (plan.get("steps") or []) if isinstance(s, dict)]
    by_id = {str(s.get("id")): s for s in steps if s.get("id")}
    endnodes: Dict[str, List[int]] = {}
    sinfo: Dict[str, _StepInfo] = {}

    for sid in _topo(steps, by_id):
        text = str(by_id[sid].get("gremlin") or "")
        events = list(_EVENT_RE.finditer(text))
        head = text[:events[0].start()] if events else text
        starts: List[int] = []
        for m in _STEP_RE.finditer(head):
            starts.extend(endnodes.get(m.group(1)) or [0])
        if _SEED_RE.search(head) or not starts:
            starts.append(0)
        cur = list(dict.fromkeys(starts))
        pending_dir, on_edge = "", False
        last_edges: List[Tuple[int, int, str, str]] = []
        measures: List[tuple] = []
        dedups: List[str] = []
        agg = ""
        def _add_hop(relation_names: List[str], d: str) -> None:
            nonlocal cur, on_edge, last_edges
            d = d or "out"
            metas = [idx.rel_meta(name) for name in relation_names]
            # Candidate = (opposite type, relation key, traversal direction,
            # required type at the current position). bothE considers both sides.
            cands: List[Tuple[str, str, str, str]] = []
            for key, fs, ts in metas:
                if d == "both":
                    cands.append((ts, key, "both", fs))
                    cands.append((fs, key, "both", ts))
                elif d == "in":
                    cands.append((fs, key, "in", ts))
                else:
                    cands.append((ts, key, "out", fs))
            cands = list(dict.fromkeys(cands))
            if not types[0] and 0 in cur and cands:
                types[0] = cands[0][3]
            by_type: Dict[str, List[Tuple[str, str, str]]] = {}
            for tt, key, direction, required in cands:
                by_type.setdefault(tt, []).append((key, direction, required))
            nxt: List[int] = []
            new_edges: List[Tuple[int, int, str, str]] = []
            for tt, group in by_type.items():
                pairs = [(u, key, direction) for u in cur
                         for key, direction, required in group
                         if _type_compat(types[u], required)]
                if not pairs:
                    continue
                path = min(path_of[u] + ((key, direction),)
                           for u, key, direction in pairs)
                types.append(tt or _UNKNOWN)
                v = len(types) - 1
                path_of[v] = path
                for u, key, direction in pairs:
                    edges.append((u, v, key, direction))
                    new_edges.append((u, v, key, direction))
                nxt.append(v)
            cur = nxt or cur
            last_edges = new_edges
            on_edge = True
        for ev in events:
            if ev.group("edge") is not None:
                pending_dir = {"outE": "out", "inE": "in",
                               "bothE": "both"}[ev.group("edge")]
                labels = _QUOTED_RE.findall(ev.group("elabel") or "")
                if labels:
                    _add_hop(labels, pending_dir)
                    pending_dir = ""
                continue
            if (ev.group("relation_name") is not None
                    or ev.group("relation_list") is not None):
                if on_edge and last_edges and not pending_dir:
                    continue
                if ev.group("relation_name") is not None:
                    relation_names = [ev.group("relation_name")]
                else:
                    relation_names = _QUOTED_RE.findall(
                        ev.group("relation_list") or "")
                if relation_names:
                    _add_hop(relation_names, pending_dir)
                    pending_dir = ""
                continue
            if ev.group("varr") is not None:
                on_edge = False
                continue
            if ev.group("val") is not None:
                attr = ev.group("val")
                if on_edge and last_edges:
                    for su, sv, rel, d in last_edges:
                        measures.append(("E", rel, attr,
                                         _path_str(path_of[su] + ((rel, d),))))
                else:
                    for u in cur:
                        measures.append(("V", types[u], attr, _path_str(path_of[u])))
                continue
            if ev.group("agg") is not None:
                agg = ev.group("agg")
                continue
            # dedup
            if on_edge and last_edges:
                for su, sv, rel, d in last_edges:
                    dedups.append(_path_str(path_of[su] + ((rel, d),)) + "#e")
            else:
                for u in cur:
                    dedups.append(_path_str(path_of[u]) + "#v")
        endnodes[sid] = cur
        sinfo[sid] = _StepInfo(tuple(measures), tuple(dedups), agg,
                               on_edge, tuple(last_edges))
    tg = _TG(tuple(types), tuple(edges))
    return tg, _stat(f, plan, tg, path_of, endnodes, sinfo)


def _kappa(f: Any) -> str:
    if not isinstance(f, dict):
        return ""
    return str(f.get("expected_value_type") or f.get("value_type") or "").lower()


def _stat(f, plan, tg, path_of, endnodes, sinfo):
    final = (plan or {}).get("final") or {}
    ft = str(final.get("type") or "").strip().lower()
    types = tg.types

    def measure_of(sid: str) -> tuple:
        info = sinfo.get(sid)
        if info is None:
            return ()
        if info.measures:
            return tuple(info.measures)
        if info.end_on_edge and info.last_edges:
            return tuple(("Ec", rel, d, _path_str(path_of[su] + ((rel, d),)))
                         for su, sv, rel, d in info.last_edges)
        return tuple(("Vc", types[u], _path_str(path_of[u]))
                     for u in (endnodes.get(sid) or [0]))

    def side_op(sid: str, mode: str) -> str:
        if str(mode).lower() == "count":
            return "count"
        info = sinfo.get(sid)
        return (info.agg if info else "") or "identity"

    if ft == "ratio":
        ns = str(final.get("numerator_step") or "")
        ds = str(final.get("denominator_step") or "")
        nm = str(final.get("numerator_mode") or "")
        dm = str(final.get("denominator_mode") or "")
        o = ("ratio", side_op(ns, nm), side_op(ds, dm))
        mu = (measure_of(ns), measure_of(ds))
    elif ft == "reduce":
        o = ("reduce", str(final.get("op") or "").strip().lower())
        mu = measure_of(str(final.get("step") or ""))
    elif ft in ("scalar", "count", "exists"):
        sid = str(final.get("step") or "")
        if ft == "scalar":
            info = sinfo.get(sid)
            o = ("scalar", (info.agg if info else "") or "identity")
        else:
            o = (ft,)
        mu = measure_of(sid)
    elif final.get("op"):
        o, mu = (str(final.get("op")).strip().lower(),), ()
    else:
        o, mu = ("",), ()
    delta = tuple(sorted({d for info in sinfo.values() for d in info.dedups}))
    return (o, mu, delta, _kappa(f))


def build_graph(f: Any, idx: SchemaIndex) -> _TG:
    return _extract(f, idx)[0]


def stat_signature(f: Any, idx: SchemaIndex) -> tuple:
    return _extract(f, idx)[1]


def _adj(tg: _TG):
    adj: List[Dict[int, list]] = [dict() for _ in tg.types]
    for u, v, r, d in tg.edges:
        adj[u].setdefault(v, []).append((r, d))
    return adj


def _slot_cost(aa, ab, u, w, up, wp) -> int:
    if u == w:
        pa = [(r, d) for r, d in aa[u].get(u, ())]
        pb = [(r, d) for r, d in ab[up].get(up, ())]
    else:
        pa = ([(r, d, ">") for r, d in aa[u].get(w, ())]
              + [(r, d, "<") for r, d in aa[w].get(u, ())])
        pb = ([(r, d, ">") for r, d in ab[up].get(wp, ())]
              + [(r, d, "<") for r, d in ab[wp].get(up, ())])
    ca, cb = Counter(pa), Counter(pb)
    inter = ca & cb
    return max(sum((ca - inter).values()), sum((cb - inter).values()))


def graph_edit_distance(a: _TG, b: _TG) -> int:
    if a == b:
        return 0
    na, nb = len(a.types), len(b.types)
    omega = (na - 1 + len(a.edges)) + (nb - 1 + len(b.edges))
    aa, ab = _adj(a), _adj(b)
    pen = 0 if a.types[0] == b.types[0] else 1
    best = [pen + omega]
    visits = [0]
    deg = [0] * na
    for u in range(na):
        deg[u] = sum(len(vs) for vs in aa[u].values())
        for w in range(na):
            if w != u:
                deg[u] += len(aa[w].get(u, ()))
    order = sorted(range(1, na), key=lambda u: (-deg[u], a.types[u], u))
    map_a: List[Any] = [None] * na
    map_a[0] = 0
    used_b = {0}

    def dfs(i: int, cost: int) -> None:
        visits[0] += 1
        if visits[0] > _GED_VISIT_CAP or cost >= best[0]:
            return
        if i == len(order):
            ins = {v for v in range(1, nb) if v not in used_b}
            c = cost + len(ins)
            if ins:
                c += sum(1 for x, y, _r, _d in b.edges if x in ins or y in ins)
            if c < best[0]:
                best[0] = c
            return
        u = order[i]
        processed = order[:i] + [0]
        c_del = cost + 1 + len(aa[u].get(u, ()))
        for w in processed:
            c_del += len(aa[u].get(w, ())) + len(aa[w].get(u, ()))
        map_a[u] = -1
        dfs(i + 1, c_del)
        map_a[u] = None
        for v in range(1, nb):
            if v in used_b or b.types[v] != a.types[u]:
                continue
            c = cost + _slot_cost(aa, ab, u, u, v, v)
            for w in processed:
                mw = map_a[w]
                if mw == -1:
                    c += len(aa[u].get(w, ())) + len(aa[w].get(u, ()))
                else:
                    c += _slot_cost(aa, ab, u, w, v, mw)
            map_a[u] = v
            used_b.add(v)
            dfs(i + 1, c)
            used_b.discard(v)
            map_a[u] = None

    dfs(0, pen)
    return best[0]


def graph_similarity(a: _TG, b: _TG) -> float:
    omega = (len(a.types) - 1 + len(a.edges)) + (len(b.types) - 1 + len(b.edges))
    if omega == 0:
        return 1.0
    return max(0.0, 1.0 - graph_edit_distance(a, b) / omega)


def _canon(tg: _TG) -> str:
    n = len(tg.types)
    groups: Dict[str, List[int]] = {}
    for u in range(1, n):
        groups.setdefault(tg.types[u], []).append(u)
    total = 1
    for g in groups.values():
        total *= factorial(len(g))
    if total > _CANON_PERM_CAP:
        return repr((tg.types, tuple(sorted(tg.edges))))
    t_order = sorted(groups)
    perms = [list(permutations(groups[t])) for t in t_order]
    best = None
    for combo in product(*perms):
        order = [0] + [u for grp in combo for u in grp]
        pos = {old: new for new, old in enumerate(order)}
        enc = tuple(sorted((pos[u], pos[v], r, d) for u, v, r, d in tg.edges))
        if best is None or enc < best:
            best = enc
    canonical_types = (tg.types[0],) + tuple(
        node_type for node_type in t_order for _ in groups[node_type])
    return repr((canonical_types, best))


def signature_key(f: Any, idx: SchemaIndex) -> str:
    tg, sig = _extract(f, idx)
    return f"{_canon(tg)}#{sig!r}"


def stat_similarity(a: tuple, b: tuple) -> float:
    got = 0.0
    for w, x, y in ((W_OP, a[0], b[0]), (W_MU, a[1], b[1]),
                    (W_DEDUP, a[2], b[2]), (W_EVT, a[3], b[3])):
        if x == y:
            got += w
    return got


def structure_similarity(a: Any, b: Any, idx: SchemaIndex) -> float:
    ta, sa = _extract(a, idx)
    tb, sb = _extract(b, idx)
    return round(W_GRAPH * graph_similarity(ta, tb)
                 + W_STAT * stat_similarity(sa, sb), 6)


def batch_metrics(features: List[Any], idx: SchemaIndex, *,
                  collect_clusters: bool = False) -> Dict[str, Any]:
    n = len(features)
    ext = [_extract(f, idx) for f in features]
    canons = [_canon(tg) for tg, _sig in ext]
    sigs = {f"{canon}#{sig!r}" for canon, (_tg, sig) in zip(canons, ext)}
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sims: List[float] = []
    for i in range(n):
        ti, si = ext[i]
        for j in range(i + 1, n):
            tj, sj = ext[j]
            graph_sim = 1.0 if canons[i] == canons[j] else graph_similarity(ti, tj)
            s = round(W_GRAPH * graph_sim + W_STAT * stat_similarity(si, sj), 6)
            sims.append(s)
            if s >= TAU:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    sizes: Dict[int, int] = {}
    for i in range(n):
        sizes[find(i)] = sizes.get(find(i), 0) + 1
    by_tau = ({f"{tau:.2f}": round(sum(1 for s in sims if s >= tau) / len(sims), 4)
               for tau in TAU_SWEEP} if sims else {f"{tau:.2f}": None
                                                   for tau in TAU_SWEEP})
    out: Dict[str, Any] = {
        "n": n, "n_pairs": len(sims),
        "distinct_structures": len(sigs),
        "dup_ratio": round(1 - len(sigs) / n, 4) if n else None,
        "avg_pairwise_sim": (round(sum(sims) / len(sims), 4) if sims else None),
        "near_dup_ratio": (round(sum(1 for s in sims if s >= TAU) / len(sims), 4)
                           if sims else None),
        "near_dup_ratio_by_tau": by_tau,
        "equivalence_classes": len(sizes),
        "redundancy_clusters": sum(1 for v in sizes.values() if v >= 2),
        "max_cluster_size": max(sizes.values()) if sizes else None,
    }
    if collect_clusters:
        groups: Dict[int, List[str]] = {}
        for i, f in enumerate(features):
            groups.setdefault(find(i), []).append(entry_key(f, i))
        out["clusters"] = [sorted(v) for v in groups.values() if len(v) >= 2]
    return out


def summarize(features: List[Dict[str, Any]], idx: SchemaIndex, *,
              pool_limit: int = 0,
              iv: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
    """Select the input prefix, then compute pool and IV-ranked top-k metrics."""
    all_features = prepare_features(features)
    pool = analysis_pool(all_features, pool_limit)
    usable = [f for f in pool if _plan_of(f) is not None]
    skipped = len(pool) - len(usable)
    if skipped:
        logger.warning("Skipping %d pool entries without Gremlin or graph_pattern metadata", skipped)
    order = ranked_indices(usable, iv)
    summary: Dict[str, Any] = {
        "features_in": len(all_features),
        "features_used": len(usable),
        "features_skipped_no_plan": skipped,
        "pool_size": len(pool),
        "pool_limit": pool_limit,
        "pool": batch_metrics(usable, idx, collect_clusters=True),
    }
    for k in (10, 30, 50, 100):
        sel = [usable[i] for i in order[:k]]
        m = batch_metrics(sel, idx) if sel else {}
        summary[f"top{k}"] = {
            "n": m.get("n"),
            "distinct_structures": m.get("distinct_structures"),
            "dup_ratio": m.get("dup_ratio"),
            "avg_pairwise_sim": m.get("avg_pairwise_sim"),
            "near_dup_ratio": m.get("near_dup_ratio"),
            "near_dup_ratio_by_tau": m.get("near_dup_ratio_by_tau"),
        }
    return summary


def render_report(summary: Dict[str, Any], *, features_path: str) -> str:
    pool = summary["pool"]
    lines = [
        "FraudFeat Structural Diversity Report (Typed Graph GED + Statistical Signature)",
        "=" * 60,
        f"Features file       : {features_path}",
        f"Input features      : {summary['features_in']}",
        f"Analysis pool size  : {summary['pool_size']}",
        f"Pool limit          : {summary['pool_limit'] or 'all'}",
        "",
        "Pool metrics:",
        f"  Distinct structures (exact signatures): {pool['distinct_structures']} / {pool['n']}",
        f"  Mean pairwise similarity             : {pool['avg_pairwise_sim']}",
        f"  Exact duplicate ratio                : {pool['dup_ratio']}",
        f"  Near-duplicate pair ratio (S>={TAU})  : {pool['near_dup_ratio']}",
        "  Threshold sweep tau=0.70..0.95       : "
        + "  ".join(f"{k}={v}" for k, v in
                   (pool.get("near_dup_ratio_by_tau") or {}).items()),
        f"  Equivalence classes / redundant clusters: {pool['equivalence_classes']} / "
        f"{pool['redundancy_clusters']} (largest={pool['max_cluster_size']})",
        "",
        "Top-k metrics (structure counts are monotonic in k and not comparable across k):",
    ]
    for k in (10, 30, 50, 100):
        st = summary.get(f"top{k}") or {}
        lines.append(f"  top{k:<4}: structures={st.get('distinct_structures')} "
                     f"mean_similarity={st.get('avg_pairwise_sim')} "
                     f"near_duplicates={st.get('near_dup_ratio')}")
    clusters = pool.get("clusters") or []
    if clusters:
        lines += ["", f"Redundant clusters (connected components at S>={TAU}, first 10):"]
        for i, c in enumerate(sorted(clusters, key=len, reverse=True)[:10], 1):
            lines.append(f"  {i}. ({len(c)}) {', '.join(c)}")
    return "\n".join(lines) + "\n"


def _load_features(path: str | Path) -> List[Dict[str, Any]]:
    return load_features(path)


def _load_iv(path: Optional[str]) -> Dict[str, float]:
    return load_iv_jsonl(path)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m fraudfeat.diversity",
        description="FraudFeat structural diversity using typed-graph GED and statistical signatures")
    ap.add_argument("--features", required=True, help="Feature JSON or JSONL file")
    ap.add_argument("--schema", required=True,
                    help="Public schema.json or its data directory for relation endpoint facts")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--pool-limit", type=int, default=100,
                    help="Take the first N entries as the analysis pool; 0 means all")
    ap.add_argument("--iv-jsonl", default=None,
                    help="effectiveness.jsonl used only for within-pool top-k IV ranking")
    ap.add_argument("--pairs", action="store_true", help="Also write pairwise similarity details")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    features = _load_features(args.features)
    if not features:
        logger.error("No evaluable features found in %s", args.features)
        return 2
    idx = SchemaIndex.load(args.schema)
    iv = _load_iv(args.iv_jsonl)
    summary = summarize(features, idx, pool_limit=args.pool_limit, iv=iv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diversity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render_report(summary, features_path=str(args.features))
    (out_dir / "diversity_report.txt").write_text(report, encoding="utf-8")
    if args.pairs:
        pair_features = [f for f in analysis_pool(features, args.pool_limit)
                         if _plan_of(f) is not None]
        with (out_dir / "diversity_pairs.jsonl").open("w", encoding="utf-8") as fp:
            for i in range(len(pair_features)):
                for j in range(i + 1, len(pair_features)):
                    fp.write(json.dumps({
                        "a": entry_key(pair_features[i], i),
                        "b": entry_key(pair_features[j], j),
                        "s": structure_similarity(pair_features[i], pair_features[j], idx),
                    }, ensure_ascii=False) + "\n")
    if not args.quiet:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
