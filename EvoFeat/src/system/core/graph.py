from __future__ import annotations

import re
from dataclasses import dataclass, field
from ..utils import get_logger
from .vocabulary import (canonical_entity, canonical_relation_name, entity_category,
                         entity_definitions, entity_label, is_public_entity,
                         public_entity_fields, public_relation,
                         public_relation_bindings, public_relation_endpoints,
                         relation_definitions)


logger = get_logger("graph")
CORE_STRONG_TYPES = {"bank_card", "phone_number"}


def parse_amount_annotation(jf: str) -> tuple[str, int | None, int | None]:
    base = jf
    total_amount = None
    trade_count = None
    idx = jf.find("(")
    if idx != -1:
        base = jf[:idx].strip()
        tail = jf[idx + 1:]
        m_amt = re.search(r"\bamount\s*=\s*(\d+)\s*fen\b", tail)
        if m_amt:
            total_amount = int(m_amt.group(1))
        m_cnt = re.search(r"\bcount\s*=\s*(\d+)\b", tail)
        if m_cnt:
            trade_count = int(m_cnt.group(1))
    return base, total_amount, trade_count


def parse_top_k_annotation(jf: str) -> int | None:
    match = re.search(r"\btop_k\s*=\s*(\d+)\b", jf)
    return int(match.group(1)) if match else None


def node_entity(n: dict) -> str:
    return canonical_entity(n.get("entity") or n.get("vtype")
                            or n.get("entity_type") or n.get("type") or "unknown")


_JOIN_FIELD_INDEX: dict[str, dict] | None = None
_ENTITY_INDEX: dict[str, dict] | None = None


def relation_info(jf: str) -> dict:
    public = public_relation(jf)
    if not public:
        return {}
    source, target = public_relation_endpoints(jf)
    return {
        "rel_id": public["en"], "name": public["en"], "en": public["en"],
        "from": source, "to": target,
        "from_short": entity_label(source), "to_short": entity_label(target),
        "from_entity": source, "to_entity": target,
        "direction": "out", "is_trade": bool(public["is_trade"]),
    }


def _relation_index() -> dict[str, dict]:
    global _JOIN_FIELD_INDEX
    if _JOIN_FIELD_INDEX is None:
        _JOIN_FIELD_INDEX = {
            join_field: relation_info(join_field)
            for fields in public_relation_bindings().values()
            for join_field in fields
        }
        _JOIN_FIELD_INDEX.update({
            name: relation_info(name) for name in relation_definitions()
        })
    return _JOIN_FIELD_INDEX


def entity_index() -> dict[str, dict]:
    global _ENTITY_INDEX
    if _ENTITY_INDEX is None:
        _ENTITY_INDEX = {
            entity: {"short": entity, "name": entity,
                     "category": entity_category(entity),
                     "fields": sorted(public_entity_fields(entity))}
            for entity in entity_definitions()
        }
    return _ENTITY_INDEX


def _resolve_relation(rel: str) -> dict | None:
    return relation_info(rel) or None


def is_account(n: dict) -> bool:
    return entity_category(node_entity(n)) == "Account (account)"


@dataclass
class Subgraph:
    subgraph_id: str
    center: str
    center_vid: str = ""
    nodes: dict[str, dict] = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)
    adj: dict[str, list[dict]] = field(default_factory=dict)
    vid_of: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Subgraph":
        if "anonymized_english_subgraph" in d:
            return cls._from_example(d)
        g = cls(subgraph_id=d.get("subgraph_id", "unknown"),
                center=d.get("center", ""),
                center_vid=d.get("center_vid", ""),
                nodes=d.get("nodes", {}),
                edges=d.get("edges", []),
                vid_of=d.get("vid_of", {}))
        g._validate_public_contract()
        g._build_adj()
        return g

    def _validate_public_contract(self) -> None:
        for node_id, node in self.nodes.items():
            entity = node_entity(node)
            if not is_public_entity(entity):
                raise ValueError(f"Unknown public entity at node {node_id}: {entity}")
            node["entity"] = entity
            node["entity_type"] = entity_label(entity)
            attributes = node.get("attributes") or {}
            unknown = sorted(set(attributes) - public_entity_fields(entity))
            if unknown:
                raise ValueError(
                    f"Unknown public attributes at node {node_id}: {', '.join(unknown)}")
        for edge in self.edges:
            relation = edge.get("join_field") or edge.get("relation") or edge.get("rel") or ""
            public = public_relation(relation)
            if not public:
                raise ValueError(f"Unknown public relation: {relation}")
            if not edge.get("join_field"):
                edge["join_field"] = canonical_relation_name(relation)
            source = node_entity(self.nodes.get(edge.get("src"), {}))
            target = node_entity(self.nodes.get(edge.get("dst"), {}))
            if (source, target) != public_relation_endpoints(relation):
                raise ValueError(
                    f"Relation {public['en']} requires {public_relation_endpoints(relation)}, "
                    f"got {(source, target)}")
            if not public["is_trade"]:
                for key in ("total_amount", "trade_count", "top_k"):
                    if edge.get(key) is not None:
                        raise ValueError(f"Non-trade relation {public['en']} cannot contain {key}")
                    edge.pop(key, None)

    @classmethod
    def _from_example(cls, d: dict) -> "Subgraph":
        aes = d.get("anonymized_english_subgraph", {})
        anon = d.get("mappings", {}).get("anonymization_mapping", {})
        tgt = aes.get("target_account", {})
        center_code = tgt.get("id", "")
        center_vid = anon.get(center_code, center_code)
        vid_of = dict(anon)
        import re as _re
        nodes: dict[str, dict] = {}
        code2entity: dict[str, str] = {}
        code2attrs: dict[str, dict] = {}
        for item in aes.get("member_attributes", []) or []:
            eid = item.get("entity")
            for code, attrs in (item.get("members", {}) or {}).items():
                code2entity[code] = eid
                code2attrs[code] = attrs
        for item in aes.get("node_summary", []) or []:
            eid = item.get("entity")
            members = item.get("members", "")
            codes: list[str] = []
            for member in (members if isinstance(members, list) else str(members).split(",")):
                member = str(member).strip()
                if "~" in member:
                    a, b = member.split("~", 1)
                    ma = _re.fullmatch(r"([A-Za-z]+)(\d+)", a)
                    mb = _re.fullmatch(r"([A-Za-z]+)(\d+)", b)
                    if ma and mb and ma.group(1) == mb.group(1):
                        pre, n0, n1 = ma.group(1), int(ma.group(2)), int(mb.group(2))
                        codes.extend(f"{pre}{i:0{len(ma.group(2))}d}" for i in range(n0, n1 + 1))
                elif member:
                    codes.append(member)
            for code in codes:
                if code not in code2entity and eid:
                    code2entity[code] = eid
        for code, eid in code2entity.items():
            nodes[code] = {"entity": eid, "first_hop": None,
                           "attributes": dict(code2attrs.get(code, {}))}
        if center_code not in code2entity:
            code2entity[center_code] = tgt.get("entity", "")
        nodes.setdefault(center_code, {"entity": code2entity[center_code],
                                        "first_hop": 0, "attributes": {}})
        nodes[center_code]["entity"] = code2entity[center_code] or tgt.get("entity", "")
        nodes[center_code]["first_hop"] = 0
        if tgt.get("attributes"):
            nodes[center_code]["attributes"] = tgt["attributes"]
        edges: list[dict] = []
        relation_mapping = d.get("mappings", {}).get("relation_mapping", {}) or {}
        for edge in aes.get("relations", []) or []:
            for key, jfs in edge.items():
                if "->" not in key:
                    continue
                src, dst = key.split("->", 1)
                for jf in (jfs or []):
                    base_jf, amt, cnt = parse_amount_annotation(jf)
                    if not public_relation(base_jf) and relation_mapping.get(base_jf):
                        base_jf = canonical_relation_name(relation_mapping[base_jf])
                    edges.append({"src": src, "dst": dst, "join_field": base_jf,
                                  "total_amount": amt, "trade_count": cnt,
                                  "top_k": parse_top_k_annotation(jf)})
        g = cls(subgraph_id=d.get("index", d.get("target_id", "unknown")),
                center=center_code, center_vid=center_vid,
                nodes=nodes, edges=edges, vid_of=vid_of)
        g._validate_public_contract()
        g._build_adj(map_rel=relation_mapping)
        g._compute_first_hop()
        return g

    def _compute_first_hop(self):
        if not self.center or self.center not in self.nodes:
            return
        from collections import deque
        dist = {self.center: 0}
        q = deque([self.center])
        while q:
            u = q.popleft()
            for e in self.adj.get(u, []):
                v = e["nbr"]
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for nid, dd in dist.items():
            if self.nodes.get(nid, {}).get("first_hop") is None:
                self.nodes[nid]["first_hop"] = dd

    def _build_adj(self, map_rel: dict | None = None):
        self.adj = {nid: [] for nid in self.nodes}
        map_rel = map_rel or {}
        for e in self.edges:
            s, t = e.get("src") or e.get("s"), e.get("dst") or e.get("t")
            if s in self.nodes and t in self.nodes:
                rel = e.get("join_field") or e.get("relation") or e.get("rel")
                info = _resolve_relation(rel) if rel else None
                canonical_rel = canonical_relation_name(rel)
                public = public_relation(rel)
                rel_id = (info or {}).get("rel_id") or (public or {}).get("en") or canonical_rel
                rel_name = (public or {}).get("en") or map_rel.get(rel) or (info or {}).get("name") or canonical_rel
                self.adj[s].append({"nbr": t, "rel": canonical_rel,
                                    "rel_id": rel_id, "rel_name": rel_name,
                                    "ts": e.get("ts"), "amt": e.get("amt"),
                                    "total_amount": e.get("total_amount"),
                                    "trade_count": e.get("trade_count"),
                                    "top_k": e.get("top_k")})
                self.adj[t].append({"nbr": s, "rel": canonical_rel,
                                    "rel_id": rel_id, "rel_name": rel_name,
                                    "ts": e.get("ts"), "amt": e.get("amt"),
                                    "total_amount": e.get("total_amount"),
                                    "trade_count": e.get("trade_count"),
                                    "top_k": e.get("top_k")})

    def neighbors(self, node: str) -> list[dict]:
        return self.adj.get(node, [])

    def edge_between(self, a: str, b: str) -> dict | None:
        for e in self.adj.get(a, []):
            if e["nbr"] == b:
                return e
        return None

    def restrict(self, visited: set[str]) -> "Subgraph":
        keep = set(visited)
        keep.add(self.center)
        nodes = {nid: nd for nid, nd in self.nodes.items() if nid in keep}
        edges = [e for e in self.edges
                 if (e.get("src") or e.get("s")) in keep
                 and (e.get("dst") or e.get("t")) in keep]
        g = Subgraph(subgraph_id=self.subgraph_id, center=self.center,
                     center_vid=self.center_vid, nodes=nodes, edges=edges,
                     vid_of=dict(self.vid_of))
        g._build_adj()
        g._compute_first_hop()
        return g

    def expand_neighbors(self, node: str, edge_type: str | None,
                         top_k: int) -> list[dict]:
        nbrs = self.neighbors(node)
        if edge_type:
            nbrs = [n for n in nbrs if n["rel"] == edge_type]
        return nbrs[:top_k]

    def amount_observations(self, center: str | None = None) -> dict:
        c = center or self.center
        out_sum = in_sum = max_single = 0.0
        cnt = 0
        peers: set[str] = set()
        for e in self.edges:
            amt = e.get("total_amount")
            if amt is None:
                continue
            tc = e.get("trade_count") or 1
            cnt += tc
            max_single = max(max_single, float(amt))
            if e.get("src") == c:
                out_sum += float(amt)
                peers.add(e.get("dst"))
            elif e.get("dst") == c:
                in_sum += float(amt)
                peers.add(e.get("src"))
        return {
            "has_amount": cnt > 0,
            "out_sum": out_sum, "in_sum": in_sum,
            "max_single": max_single, "count": cnt,
            "counterparties": sorted(peers),
        }

    def detect_signals(self) -> list[dict]:
        signals = []
        for mid, attrs in self.nodes.items():
            if node_entity(attrs) in CORE_STRONG_TYPES:
                connected_accounts = [e["nbr"] for e in self.adj.get(mid, [])
                                      if is_account(self.nodes.get(e["nbr"], {}))]
                if len(connected_accounts) >= 2:
                    black_cnt = sum(1 for a in connected_accounts
                                    if self.nodes.get(a, {}).get("label") == "black")
                    signals.append({
                        "signal": "shared_strong_field", "field": node_entity(attrs),
                        "medium": mid, "accounts": connected_accounts,
                        "group_size": len(connected_accounts), "black_in_group": black_cnt,
                    })
        for mid, attrs in self.nodes.items():
            if is_account(attrs):
                continue
            shared = [e["nbr"] for e in self.adj.get(mid, [])
                      if is_account(self.nodes.get(e["nbr"], {}))
                      and self._is_shared_identity(e["rel"])]
            if len(shared) >= 2:
                signals.append({
                    "signal": "shared_strong_field", "field": node_entity(attrs),
                    "medium": mid, "accounts": shared,
                    "group_size": len(shared), "black_in_group": 0,
                })
        for nid, attrs in self.nodes.items():
            if not is_account(attrs):
                continue
            out = [e for e in self.adj.get(nid, []) if e["nbr"] != nid]
            if len(out) >= 3:
                signals.append({
                    "signal": "fan_out", "center": nid,
                    "downstream": [e["nbr"] for e in out], "count": len(out),
                })
        for nid in self.nodes:
            peers = set()
            for e in self.adj.get(nid, []):
                rel = e["rel"] or ""
                if "transfer" in rel or "payment" in rel.lower() or self._is_mutual(e):
                    peers.add(e["nbr"])
            for p in peers:
                back = any((e["nbr"] == nid)
                           for e in self.adj.get(p, []))
                if back:
                    signals.append({"signal": "mutual_transfer", "a": nid, "b": p})
                    break
        return signals

    @staticmethod
    def _is_shared_identity(rel: str | None) -> bool:
        if not rel:
            return False
        rel = str(rel).lower()
        return any(k in rel for k in ("cre_id", "customer_id", "account_uin",
                                      "same", "find_sys_no"))

    @staticmethod
    def _is_mutual(edge: dict) -> bool:
        return False
