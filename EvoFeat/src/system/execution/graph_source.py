from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from ..core.graph import Subgraph
from ..core.graph_clean import GraphCleaner
from ..paths import DATA_DIR
from ..utils import get_logger
from .eg_plan import query_mode, sanitize_vid, vid_to_token
from ..core.vocabulary import (canonical_entity, entity_definitions, entity_prefix,
                               public_relation, public_relation_endpoints)


logger = get_logger("execution.graph_source")
_STRONG_MEDIA = ("bank_card", "phone_number", "natural_person", "individual_customer")
_PLACEHOLDER_TB = "__from_relation__"
_DATE_SUFFIX_RE = re.compile(r"_\d{4}_\d{4}$")


def _prefix_for(entity: str) -> str:
    return entity_prefix(entity)


def _unwrap(v: Any) -> Any:
    if isinstance(v, list):
        return v[0] if v else None
    return v


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _to_optional_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    return int(_to_float(v))


def _chunks(lst: Sequence, n: int) -> Iterable[Sequence]:
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


_Q_VERTICES_TPL = ("{g}.V(ids)"
                   ".project('id','label','props')"
                   ".by(id).by(label).by(valueMap()).toList()")
_Q_EDGES_TPL = ("{g}.V(ids).bothE().dedup().limit({k})"
                ".project('id','props','src','dst')"
                ".by(id).by(valueMap()).by(outV().id()).by(inV().id()).toList()")


class StaticSource:
    live = False
    min_hops = 0

    def prepare(self, sg: Subgraph, state) -> bool:
        return True

    def ensure_expanded(self, sg: Subgraph, ids: Iterable[str]) -> None:
        return None

    def finalize(self, sg: Subgraph, visited: Set[str]) -> None:
        return None

    def is_pending(self, sid: str) -> bool:
        return False

    def materialize_pending(self, sg: Subgraph, sids: Sequence[str]) -> Tuple[List[str], List[str]]:
        return [], []

    def pick_groups(self, sg: Subgraph, spec: Sequence[dict]) -> Tuple[List[str], str]:
        return [], ""

    def discard_groups(self, keys: Sequence[str]) -> int:
        return 0

    def pending_view(self, sg: Subgraph) -> str:
        return ""


class LiveGraphSource:
    live = True

    def __init__(self, executor, cleaner: GraphCleaner | None = None, *,
                 per_hop_caps: Sequence[int] = (50, 250, 500, 1000, 2000),
                 edge_limit: int = 5000, min_hops: int = 3,
                 probe_min_hops: int | None = None,
                 probe_max_frontier: int = 8, induced_edges: bool = True,
                 entity_names: Sequence[str] | None = None,
                 query_chunk: int = 200, probe_quiet: bool = True):
        self.executor = executor
        self.cleaner = cleaner or GraphCleaner()
        self.per_hop_caps = [int(x) for x in per_hop_caps]
        self.edge_limit = int(edge_limit)
        self.min_hops = int(min_hops)
        self.probe_min_hops = self.min_hops if probe_min_hops is None else int(probe_min_hops)
        self.probe_max_frontier = int(probe_max_frontier)
        self.induced_edges = bool(induced_edges)
        self._entity_names = sorted(entity_names or [], key=len, reverse=True)
        self.query_chunk = int(query_chunk)
        self.probe_quiet = bool(probe_quiet)
        self._reset()

    def _reset(self) -> None:
        self._vid2sid: Dict[str, str] = {}
        self._tokens: Dict[str, str] = {}
        self._cands: Dict[str, Dict[str, Any]] = {}
        self._sid2cand: Dict[str, Dict[str, Any]] = {}
        self._counters: Counter = Counter()
        self._fetched: Set[str] = set()
        self._edge_seen: Dict[tuple, Tuple[str, tuple]] = {}
        self._group_map: Dict[str, List[str]] = {}
        self._stats: Counter = Counter()
        self._prepared_ok: bool | None = None

    def _q_vertices(self, tokens: Sequence[str]) -> List[Dict[str, Any]]:
        if query_mode() == "binding":
            q = _Q_VERTICES_TPL.format(g=self.executor.g_prefix)
            rows = self.executor.submit(q, {"ids": list(tokens)})
        else:
            id_list = ",".join(json.dumps(t, ensure_ascii=False) for t in tokens)
            q = (f"{self.executor.g_prefix}.V({id_list})"
                 ".project('id','label','props').by(id).by(label).by(valueMap()).toList()")
            rows = self.executor.submit(q, {})
        return rows if isinstance(rows, list) else []

    def _q_edges(self, tokens: Sequence[str]) -> List[Dict[str, Any]]:
        if query_mode() == "binding":
            q = _Q_EDGES_TPL.format(g=self.executor.g_prefix, k=self.edge_limit)
            rows = self.executor.submit(q, {"ids": list(tokens)})
        else:
            id_list = ",".join(json.dumps(t, ensure_ascii=False) for t in tokens)
            q = (f"{self.executor.g_prefix}.V({id_list}).bothE().dedup()"
                 f".limit({self.edge_limit})"
                 ".project('id','props','src','dst')"
                 ".by(id).by(valueMap()).by(outV().id()).by(inV().id()).toList()")
            rows = self.executor.submit(q, {})
        return rows if isinstance(rows, list) else []

    def _alloc_sid(self, entity: str) -> str:
        prefix = _prefix_for(entity)
        self._counters[prefix] += 1
        return f"{prefix}{self._counters[prefix]:03d}"

    def _resolve_entity(self, tb: str, vtype: str, vid: str) -> str:
        for cand in (vtype, tb):
            value = _DATE_SUFFIX_RE.sub("", (cand or "").strip())
            entity = canonical_entity(value)
            if (entity in entity_definitions() and value != _PLACEHOLDER_TB
                    and value.lower() != "unknown"):
                return entity
        for name in self._entity_names:
            if vid == name or vid.startswith(name + "_"):
                entity = canonical_entity(name)
                if entity in entity_definitions():
                    return entity
        return ""

    def _parse_attrs(self, props: Dict[str, Any]) -> Dict[str, Any]:
        raw = _unwrap((props or {}).get("attributes"))
        if isinstance(raw, dict):
            return raw
        if not raw:
            return {}
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except (TypeError, ValueError):
            return {}

    @staticmethod
    def _make_token(vid: str) -> str:
        return vid_to_token(vid)

    def _cap(self, depth: int) -> int:
        idx = depth - 1
        if idx < len(self.per_hop_caps):
            return self.per_hop_caps[idx]
        return self.per_hop_caps[-1] if self.per_hop_caps else 1000

    def _depth_count(self, depth: int) -> int:
        return sum(1 for c in self._cands.values() if c["materialized"] and c["depth"] <= depth)

    def _depth_left(self, depth: int) -> int:
        return max(0, self._cap(depth) - self._depth_count(depth))

    def init_target(self, vid: str, entity_hint: str = "") -> Subgraph | None:
        self._reset()
        vid = str(vid).strip()
        if not vid:
            return None
        token = vid_to_token(vid)
        try:
            rows = self._q_vertices([token])
        except Exception as e:
            logger.warning(f"[live graph] Seed vertex query failed {vid[:50]}: {e}")
            return None
        if not rows:
            logger.info(f"[live graph] Seed vertex not found; skipping: {vid[:60]}")
            return None
        row = rows[0]
        canon = str(row.get("id") or sanitize_vid(vid))
        props = row.get("props") or {}
        tb = str(_unwrap(props.get("source_table")) or "")
        vtype = str(_unwrap(props.get("vtype")) or _unwrap(props.get("entity_type")) or "")
        entity = self._resolve_entity(tb, vtype, canon)
        if not entity:
            hinted = canonical_entity(entity_hint)
            entity = hinted if hinted in entity_definitions() else ""
        if not entity or self.cleaner.vtype_blocked(entity):
            logger.info("[live graph] Target entity is outside the public vocabulary; skipping")
            return None
        self._tokens[canon] = vid_to_token(canon)
        attrs = self.cleaner.clean_attrs(entity, self._parse_attrs(props))
        sid = self._alloc_sid(entity)
        cand = self._new_cand(canon, tb, entity, depth=0, sid=sid)
        cand["materialized"] = True
        cand["attrs"] = attrs
        sg = Subgraph(
            subgraph_id=canon, center=sid, center_vid=canon,
            nodes={sid: {"entity": entity, "first_hop": 0, "attributes": attrs}},
            edges=[], vid_of={sid: canon})
        sg._build_adj()
        logger.info(f"[live graph] Target initialized: {sid}={canon[:60]} entity={entity}")
        return sg

    def _new_cand(self, vid: str, tb: str, entity: str, depth: int, sid: str | None = None) -> Dict[str, Any]:
        sid = sid or self._alloc_sid(entity)
        cand: Dict[str, Any] = {
            "vid": vid, "sid": sid, "entity": entity, "tb": tb,
            "attrs": {}, "depth": depth,
            "srcs": set(), "rels": set(), "amt_max": 0.0, "txn": 0,
            "recs": [], "materialized": False, "dead": False,
        }
        self._cands[vid] = cand
        self._sid2cand[sid] = cand
        self._vid2sid[vid] = sid
        self._tokens[vid] = self._make_token(vid)
        return cand

    def _eg_log_silence(self) -> Tuple[logging.Logger, int]:
        eg_log = logging.getLogger("system.execution.eg")
        prev = eg_log.level
        if self.probe_quiet:
            eg_log.setLevel(logging.WARNING)
        return eg_log, prev

    @staticmethod
    def _eg_log_restore(eg_log: logging.Logger, prev: int) -> None:
        eg_log.setLevel(prev)

    def prepare(self, sg: Subgraph, state) -> bool:
        if self._prepared_ok is not None:
            return self._prepared_ok
        eg_log, prev = self._eg_log_silence()
        try:
            self._prepared_ok = self._prepare_inner(sg)
        finally:
            self._eg_log_restore(eg_log, prev)
        return self._prepared_ok

    def _prepare_inner(self, sg: Subgraph) -> bool:
        center_vid = sg.center_vid
        self._fetch_neighborhood(sg, center_vid)
        for d in range(2, self.probe_min_hops + 1):
            if self._max_depth(sg) >= self.probe_min_hops:
                return True
            frontier = [c for c in self._cands.values()
                        if c["materialized"] and not c["dead"]
                        and c["depth"] == d - 1 and c["vid"] not in self._fetched]
            if not frontier:
                pend = [c for c in self._cands.values()
                        if not c["materialized"] and not c["dead"] and c["depth"] == d - 1]
                if pend:
                    pend.sort(key=lambda c: (-len(c["srcs"]), -c["amt_max"], c["vid"]))
                    self._materialize(sg, [c["vid"] for c in pend[:self.probe_max_frontier]],
                                      ignore_cap=True)
                    frontier = [c for c in self._cands.values()
                                if c["materialized"] and not c["dead"]
                                and c["depth"] == d - 1 and c["vid"] not in self._fetched]
            if not frontier:
                break
            frontier.sort(key=lambda c: (-len(c["srcs"]), -c["amt_max"], c["vid"]))
            for c in frontier[:self.probe_max_frontier]:
                self._fetch_neighborhood(sg, c["vid"])
        reached = self._max_depth(sg)
        if reached < self.probe_min_hops:
            logger.info(f"[live graph] Target reaches {reached} hops < {self.probe_min_hops}; skipping shallow graph "
                        f"(stats={dict(self._stats)})")
            return False
        logger.info(f"[live graph] Depth probe passed (reachable depth {reached}); starting LLM exploration "
                    f"(nodes={len(sg.nodes)}, edges={len(sg.edges)}; probe state discarded and exploration data refetched)")

        self._discard_probe_state(sg)
        self._fetch_neighborhood(sg, center_vid)
        return True

    def _discard_probe_state(self, sg: Subgraph) -> None:
        center_sid, center_vid = sg.center, sg.center_vid
        center_node = sg.nodes.get(center_sid) or {}
        center_cand = self._cands.get(center_vid)
        stats = self._stats
        center_token = self._tokens.get(center_vid)
        self._reset()
        self._stats = stats
        if center_token:
            self._tokens[center_vid] = center_token
        if center_cand is not None:
            center_cand["srcs"] = set()
            center_cand["rels"] = set()
            center_cand["amt_max"] = 0.0
            center_cand["txn"] = 0
            center_cand["recs"] = []
            self._cands[center_vid] = center_cand
            self._sid2cand[center_sid] = center_cand
            self._vid2sid[center_vid] = center_sid
            self._counters[_prefix_for(center_cand["entity"])] = 1
        sg.nodes = {center_sid: center_node}
        sg.edges = []
        sg.vid_of = {center_sid: center_vid}
        sg._build_adj()
        sg._compute_first_hop()

    def _max_depth(self, sg: Subgraph) -> int:
        return max((n.get("first_hop", 0) for n in sg.nodes.values()), default=0)

    def ensure_expanded(self, sg: Subgraph, ids: Iterable[str]) -> None:
        eg_log, prev = self._eg_log_silence()
        try:
            for sid in ids:
                vid = sg.vid_of.get(sid)
                if vid:
                    self._fetch_neighborhood(sg, vid)
        finally:
            self._eg_log_restore(eg_log, prev)

    def _fetch_neighborhood(self, sg: Subgraph, vid: str) -> None:
        if vid in self._fetched:
            return
        self._fetched.add(vid)
        token = self._tokens.get(vid)
        if not token:
            return
        src_cand = self._cands.get(vid)
        src_depth = src_cand["depth"] if src_cand else 0
        try:
            rows = self._q_edges([token])
        except Exception as e:
            logger.warning(f"[live graph] Neighborhood fetch failed {vid[:40]}: {e}")
            self._stats["fetch_err"] += 1
            return
        if len(rows) >= self.edge_limit:
            self._stats["truncated"] += 1
            logger.warning(f"[live graph] {vid[:40]} reached the edge safety limit {self.edge_limit}; truncated (high-degree public hub)")
        new_vids: List[str] = []
        changed = False
        for r in rows:
            props = r.get("props") or r
            jf = str(_unwrap(props.get("join_field")) or _unwrap(props.get("relation")) or "").strip()
            public = public_relation(jf)
            src = str(r.get("src") or "").strip()
            dst = str(r.get("dst") or "").strip()
            if not public:
                self._stats["rel_unknown"] += 1
                continue
            if not src or not dst or src == dst:
                continue
            other = dst if src == vid else src
            if self.cleaner.vid_blocked(other):
                self._stats["vid_bl"] += 1
                continue
            is_trade = bool(public["is_trade"])
            amt = _to_float(_unwrap(props.get("total_amount"))) if is_trade else 0.0
            cnt = int(_to_float(_unwrap(props.get("trade_count")))) if is_trade else 0
            top = _to_optional_int(_unwrap(props.get("top_k"))) if is_trade else None
            rec = {"src": src, "dst": dst, "jf": jf, "amt": amt, "cnt": cnt, "top": top}
            cand = self._cands.get(other)
            if cand is None:
                entity = self._resolve_entity("", "", other)
                if entity and self.cleaner.vtype_blocked(entity):
                    self._stats["vtype_bl"] += 1
                    continue
                cand = self._new_cand(other, "", entity, src_depth + 1)
                new_vids.append(other)
            elif cand["dead"]:
                continue
            cand["srcs"].add(vid)
            cand["rels"].add(jf)
            cand["amt_max"] = max(cand["amt_max"], amt)
            cand["txn"] += cnt
            cand["recs"].append(rec)
            if self._try_add_edge(sg, rec):
                changed = True
        self._stats["edges"] += len(rows)

        self._auto_materialize(sg, new_vids)
        if changed:
            sg._build_adj()
            sg._compute_first_hop()

    def _auto_materialize(self, sg: Subgraph, new_vids: Sequence[str]) -> None:
        by_depth: Dict[int, List[str]] = defaultdict(list)
        for v in new_vids:
            c = self._cands[v]
            if not c["dead"]:
                by_depth[c["depth"]].append(v)
        for d, vids in by_depth.items():
            left = self._depth_left(d)
            if left > 0 and len(vids) <= left:
                self._materialize(sg, vids)
            else:
                self._stats["pending"] += len(vids)

    def _materialize(self, sg: Subgraph, vids: Sequence[str], ignore_cap: bool = False) -> List[str]:
        todo = []
        for v in vids:
            c = self._cands.get(v)
            if c and not c["materialized"] and not c["dead"]:
                if not ignore_cap and self._depth_left(c["depth"]) - len(todo) <= 0:
                    continue
                todo.append(v)
        if not todo:
            return []
        props_map: Dict[str, Dict[str, Any]] = {}
        pairs = [(v, self._tokens.get(v)) for v in todo]
        pairs = [(v, t) for v, t in pairs if t]
        for chunk in _chunks(pairs, self.query_chunk):
            try:
                rows = self._q_vertices([t for _, t in chunk])
            except Exception as e:
                logger.warning(f"[live graph] Batched vertex attribute query failed: {e}")
                self._stats["props_err"] += 1
                continue
            for row in rows:
                bare = str(row.get("id") or "")
                if bare:
                    props_map[bare] = row
        done: List[str] = []
        changed = False
        for v in todo:
            cand = self._cands[v]
            row = props_map.get(v)
            if row is None:
                self._stats["props_miss"] += 1
            else:
                props = row.get("props") or {}
                tb = str(_unwrap(props.get("source_table")) or "")
                vtype = str(_unwrap(props.get("vtype")) or _unwrap(props.get("entity_type")) or "")
                entity = self._resolve_entity(tb, vtype, v)
                if not entity or self.cleaner.vtype_blocked(entity):
                    cand["dead"] = True
                    self._stats["vtype_bl"] += 1
                    continue
                cand["entity"] = entity
                cand["attrs"] = self.cleaner.clean_attrs(entity, self._parse_attrs(props))
            if not cand["entity"]:
                cand["dead"] = True
                self._stats["vtype_bl"] += 1
                continue
            sid = cand["sid"]
            if sid.startswith("N"):
                new_sid = self._alloc_sid(cand["entity"])
                del self._sid2cand[sid]
                cand["sid"] = new_sid
                self._sid2cand[new_sid] = cand
                self._vid2sid[v] = new_sid
                sid = new_sid
            sg.nodes[sid] = {"entity": cand["entity"], "first_hop": cand["depth"],
                             "attributes": cand.get("attrs") or {}}
            sg.vid_of[sid] = v
            cand["materialized"] = True
            done.append(sid)
            changed = True
        for v in todo:
            cand = self._cands[v]
            if not cand["materialized"]:
                continue
            for rec in cand["recs"]:
                if self._try_add_edge(sg, rec):
                    changed = True
        if changed:
            sg._build_adj()
            sg._compute_first_hop()
        return done

    @staticmethod
    def _fold_key(a: str, b: str, jf: str) -> tuple:
        public = public_relation(jf)
        relation = str(public.get("en") or jf)
        if public.get("is_trade"):
            return ("trade", a, b, relation)
        x, y = sorted([a, b])
        return ("association", x, y, relation)

    def _try_add_edge(self, sg: Subgraph, rec: Dict[str, Any]) -> bool:
        src_c = self._cands.get(rec["src"])
        dst_c = self._cands.get(rec["dst"])
        if not src_c or not dst_c or not src_c["materialized"] or not dst_c["materialized"]:
            return False
        public = public_relation(rec["jf"])
        actual = (src_c["entity"], dst_c["entity"])
        if not public or actual != public_relation_endpoints(rec["jf"]):
            return False
        key = self._fold_key(rec["src"], rec["dst"], rec["jf"])
        if key in self._edge_seen:
            return False
        self._edge_seen[key] = (rec["src"], ())
        edge = {"src": src_c["sid"], "dst": dst_c["sid"], "join_field": rec["jf"]}
        if public["is_trade"]:
            edge.update({"total_amount": rec["amt"], "trade_count": rec["cnt"],
                         "top_k": rec["top"]})
        sg.edges.append(edge)
        self._stats["edges_in"] += 1
        return True

    def is_pending(self, sid: str) -> bool:
        c = self._sid2cand.get(sid)
        return bool(c and not c["materialized"] and not c["dead"])

    def _pending(self) -> List[Dict[str, Any]]:
        return [c for c in self._cands.values() if not c["materialized"] and not c["dead"]]

    def _build_groups(self) -> Dict[str, List[str]]:
        buckets: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for c in self._pending():
            primary_rel = sorted(c["rels"])[0] if c["rels"] else "?"
            buckets[(c["entity"], primary_rel)].append(c)
        ordered = sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        gmap: Dict[str, List[str]] = {}
        for i, (_k, members) in enumerate(ordered, 1):
            members.sort(key=lambda c: (-len(c["srcs"]), -c["amt_max"], c["vid"]))
            gmap[f"G{i}"] = [c["vid"] for c in members]
        self._group_map = gmap
        return gmap

    def _group_meta(self) -> Dict[str, tuple]:
        buckets: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for c in self._pending():
            primary_rel = sorted(c["rels"])[0] if c["rels"] else "?"
            buckets[(c["entity"], primary_rel)].append(c)
        ordered = sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        return {f"G{i}": (k[0], k[1], members) for i, (k, members) in enumerate(ordered, 1)}

    def pending_view(self, sg: Subgraph) -> str:
        pend = self._pending()
        if not pend:
            return ""
        meta = self._group_meta()
        self._group_map = {k: [c["vid"] for c in sorted(m, key=lambda c: (-len(c["srcs"]), -c["amt_max"], c["vid"]))]
                           for k, (_e, _r, m) in meta.items()}
        depths = sorted({c["depth"] for c in pend} | {c["depth"] for c in self._cands.values() if c["materialized"] and c["depth"] > 0})
        budget_str = " ".join(f"hop {d} remaining={self._depth_left(d)}" for d in depths)
        lines = [
            f"[OVER-LIMIT CANDIDATES: GROUP SUMMARY] (pending={len(pend)}; {budget_str})",
            "Expand by group: {\"action\":\"Expand\",\"groups\":[{\"key\":\"G1\",\"top\":10,\"why\":\"...\"}]}; "
            "discard by group: {\"action\":\"Discard\",\"groups\":[\"G1\"]}; highlighted individuals can be Expanded by short ID",
        ]
        hi = [c for c in pend if len(c["srcs"]) >= 2
              or any(k in c["entity"].lower() for k in _STRONG_MEDIA)]
        hi.sort(key=lambda c: (-len(c["srcs"]), -c["amt_max"]))
        if hi:
            chips = []
            for c in hi[:15]:
                src_sids = ",".join(sorted(self._vid2sid.get(s, "?") for s in c["srcs"]))
                amt = f" {c['amt_max']:.0f} fen" if c["amt_max"] > 0 else ""
                chips.append(f"{c['sid']}({c['entity'][:24]},bridges={len(c['srcs'])},{src_sids}{amt})")
            lines.append("Individual highlights: " + " ".join(chips))
        for gkey, (entity, rel, members) in list(meta.items())[:25]:
            srcs = sorted({self._vid2sid.get(s, "?") for m in members for s in m["srcs"]})
            bridges = sum(1 for m in members if len(m["srcs"]) >= 2)
            amt_max = max((m["amt_max"] for m in members), default=0.0)
            amt_sum = sum(m["amt_max"] for m in members)
            samples = ",".join(m["sid"] for m in members[:4])
            amt_str = f" amount_sum={amt_sum:.0f} fen/max={amt_max:.0f} fen" if amt_sum > 0 else ""
            lines.append(
                f"{gkey} | {entity} x {rel} <- {','.join(srcs[:4])} | count={len(members)}"
                f" | bridges={bridges}{amt_str} | samples={samples}")
        if len(meta) > 25:
            lines.append(f"... {len(meta) - 25} additional groups omitted")
        return "\n".join(lines)

    def pick_groups(self, sg: Subgraph, spec: Sequence[dict]) -> Tuple[List[str], str]:
        eg_log, prev = self._eg_log_silence()
        try:
            return self._pick_groups_inner(sg, spec)
        finally:
            self._eg_log_restore(eg_log, prev)

    def _pick_groups_inner(self, sg: Subgraph, spec: Sequence[dict]) -> Tuple[List[str], str]:
        if not self._group_map:
            self._build_groups()
        out: List[str] = []
        notes: List[str] = []
        for item in spec[:8]:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "").strip()
            top = max(1, min(50, int(item.get("top") or 10)))
            vids = self._group_map.get(key)
            if not vids:
                notes.append(f"{key} does not exist or was already processed")
                continue
            picked: List[str] = []
            for v in vids:
                c = self._cands[v]
                if c["materialized"] or c["dead"]:
                    continue
                if self._depth_left(c["depth"]) - len(picked) <= 0:
                    break
                picked.append(v)
                if len(picked) >= top:
                    break
            sids = self._materialize(sg, picked)
            out.extend(sids)
            notes.append(f"{key}: materialized={len(sids)}")
        return out, "; ".join(notes)

    def materialize_pending(self, sg: Subgraph, sids: Sequence[str]) -> Tuple[List[str], List[str]]:
        eg_log, prev = self._eg_log_silence()
        try:
            return self._materialize_pending_inner(sg, sids)
        finally:
            self._eg_log_restore(eg_log, prev)

    def _materialize_pending_inner(self, sg: Subgraph, sids: Sequence[str]) -> Tuple[List[str], List[str]]:
        ok: List[str] = []
        rej: List[str] = []
        for sid in sids:
            c = self._sid2cand.get(sid)
            if not c or c["materialized"] or c["dead"]:
                continue
            if self._depth_left(c["depth"]) <= 0:
                rej.append(sid)
                continue
            done = self._materialize(sg, [c["vid"]])
            ok.extend(done)
        return ok, rej

    def discard_groups(self, keys: Sequence[str]) -> int:
        if not self._group_map:
            self._build_groups()
        n = 0
        for key in keys:
            for v in self._group_map.get(str(key).strip(), []):
                c = self._cands.get(v)
                if c and not c["materialized"] and not c["dead"]:
                    c["dead"] = True
                    n += 1
        return n

    def finalize(self, sg: Subgraph, visited: Set[str]) -> None:
        if not self.induced_edges:
            return
        eg_log, prev = self._eg_log_silence()
        try:
            self._finalize_inner(sg, visited)
        finally:
            self._eg_log_restore(eg_log, prev)

    def _finalize_inner(self, sg: Subgraph, visited: Set[str]) -> None:
        vids = [sg.vid_of[s] for s in visited if s in sg.vid_of]
        vset = set(vids)
        tokens = [self._tokens[v] for v in vids if self._tokens.get(v)]
        if not tokens:
            return
        added = 0
        for chunk in _chunks(tokens, self.query_chunk):
            try:
                rows = self._q_edges(list(chunk))
            except Exception as e:
                logger.warning(f"[live graph] Induced-edge query failed: {e}")
                continue
            for r in rows:
                src = str(r.get("src") or "")
                dst = str(r.get("dst") or "")

                props = r.get("props") or r
                jf = str(_unwrap(props.get("join_field")) or _unwrap(props.get("relation")) or "").strip()
                public = public_relation(jf)
                if not public:
                    self._stats["rel_unknown"] += 1
                    continue
                if src in vset and dst in vset and src != dst:
                    is_trade = bool(public["is_trade"])
                    rec = {"src": src, "dst": dst, "jf": jf,
                           "amt": _to_float(_unwrap(props.get("total_amount"))) if is_trade else 0.0,
                           "cnt": int(_to_float(_unwrap(props.get("trade_count")))) if is_trade else 0,
                           "top": _to_optional_int(_unwrap(props.get("top_k"))) if is_trade else None}
                    if self._try_add_edge(sg, rec):
                        added += 1
        if added:
            sg._build_adj()
            sg._compute_first_hop()
        logger.info(f"[live graph] Induced edges added: {added} (visited={len(vset)}) "
                    f"stats={dict(self._stats)}")


class RandomSampleSource(LiveGraphSource):
    def __init__(self, *args, seed: int = 20260825, **kw):
        super().__init__(*args, **kw)
        self._rng = random.Random(seed)

    def _auto_materialize(self, sg: Subgraph, new_vids: Sequence[str]) -> None:
        return

    def prepare(self, sg: Subgraph, state) -> bool:
        if self._prepared_ok is not None:
            return self._prepared_ok
        eg_log, prev = self._eg_log_silence()
        try:
            self._prepare_random_inner(sg)
        finally:
            self._eg_log_restore(eg_log, prev)
        reached = self._max_depth(sg)
        if self.probe_min_hops > 0 and reached < self.probe_min_hops:
            self._prepared_ok = False
            return False
        logger.info(f"[random-sampling ablation] Random subgraph expansion complete: nodes={len(sg.nodes)}, "
                    f"edges={len(sg.edges)} (random BFS, caps={self.per_hop_caps}, "
                    f"reachable depth={reached})")
        self._prepared_ok = True
        return True

    def _prepare_random_inner(self, sg: Subgraph) -> None:
        self._fetch_neighborhood(sg, sg.center_vid)
        for depth in range(1, len(self.per_hop_caps) + 1):
            candidates = [c for c in self._cands.values()
                          if not c["materialized"] and not c["dead"]
                          and c["depth"] == depth]
            if not candidates:
                break
            budget = self._depth_left(depth)
            if len(candidates) > budget:
                candidates = self._rng.sample(candidates, budget)
            selected = [c["vid"] for c in candidates]
            self._materialize(sg, selected)
            if depth == len(self.per_hop_caps):
                break
            for vid in selected:
                self._fetch_neighborhood(sg, vid)


class ParquetGraphSource(LiveGraphSource):
    """LiveGraphSource adapter over the shared read-only FraudFeat index."""

    @staticmethod
    def _python_value(value: Any) -> Any:
        if value is None or type(value).__name__ == "NAType":
            return None
        if type(value).__module__.startswith("numpy") and hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and value != value:
            return None
        if hasattr(value, "isoformat") and callable(value.isoformat):
            return value.isoformat()
        return value

    def _bare_id(self, token: str) -> str:
        value = sanitize_vid(token)
        if self.executor.store.has_vertex(value):
            return value
        biz = value.rsplit("|", 1)[0] if "|" in value else value
        return biz if self.executor.store.has_vertex(biz) else value

    def _q_vertices(self, tokens: Sequence[str]) -> List[dict]:
        rows = []
        for token in tokens:
            uid = self._bare_id(token)
            if not self.executor.store.has_vertex(uid):
                continue
            row = self.executor.store.vertex(uid)
            entity = self.executor.store.entity_of[uid]
            attrs = {}
            for key in row.index:
                if key == "entity_type":
                    continue
                value = self._python_value(row[key])
                if value is not None:
                    attrs[str(key)] = value
            rows.append({"id": uid, "label": "entity",
                         "props": {"vtype": entity, "source_table": entity,
                                   "attributes": attrs}})
        return rows

    def _q_edges(self, tokens: Sequence[str]) -> List[dict]:
        positions: Dict[int, None] = {}
        for token in tokens:
            uid = self._bare_id(token)
            for pos in self.executor.store.incident(uid):
                positions[int(pos)] = None
        selected = list(positions)[:self.edge_limit]
        frame = self.executor.store.edge_rows(selected)
        rows = []
        for pos, row in frame.iterrows():
            relation = str(row["relation"])
            props = {"relation": relation}
            for key in ("total_amount", "trade_count", "top_k"):
                value = self._python_value(row.get(key))
                if value is not None:
                    props[key] = value
            rows.append({"id": str(row.get("edge_id") or pos),
                         "src": str(row["src"]), "dst": str(row["dst"]),
                         "label": relation, "props": props})
        return rows


class ParquetRandomSampleSource(ParquetGraphSource, RandomSampleSource):
    """Random-subgraph ablation using the same Parquet backend."""


class ParquetTargetStream:
    """Deterministic positive-target stream prepared from released labels."""
    ACCOUNT = "wechat_pay_account"

    def __init__(self, source: ParquetGraphSource, *, sample: int = 100,
                 seed: int = 42, targets_path: str | Path | None = None):
        self.source = source
        self.sample = int(sample)
        self.seed = int(seed)
        if targets_path is None:
            cache = DATA_DIR / "splits"
            targets_path = cache / f"target_streams_s{seed}.json"
        self.targets_path = Path(targets_path)
        if not self.targets_path.is_file():
            raise FileNotFoundError(
                f"Official discovery targets not found: {self.targets_path}. "
                "They ship with the dataset under data/splits/.")
        from .parquet_backend import public_ids
        self._queue = list(dict.fromkeys(public_ids(self.targets_path)))
        self._cursor = 0
        self._produced = 0
        self._shallow = 0

    @property
    def probe_stats(self) -> dict:
        return {"shallow_skipped": self._shallow}

    def next(self) -> Subgraph | None:
        while self._cursor < len(self._queue) and self._produced < self.sample:
            vid = self._queue[self._cursor]
            self._cursor += 1
            sg = self.source.init_target(vid, entity_hint=self.ACCOUNT)
            if not sg.center_vid:
                continue
            if self.source.probe_min_hops > 0 and not self.source.prepare(sg):
                self._shallow += 1
                continue
            self._produced += 1
            return sg
        return None

    def close(self) -> None:
        return None
