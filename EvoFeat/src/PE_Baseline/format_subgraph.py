import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from system.core.vocabulary import (canonical_entity, canonical_relation_name,
                                    entity_category, entity_definitions,
                                    entity_prefix, is_public_entity, public_entity_fields,
                                    public_relation)
from system.paths import CONFIG_DIR, SYSTEM_ROOT


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = str(CONFIG_DIR / "fraudfeat_schema.json")
BLACKLIST_PATH = str(SYSTEM_ROOT / "data" / "graph_res" / "edge_field_blacklist.txt")
TRADE_SCORE = 100.0
ENCRYPTED_SUFFIXES = ("_encrypted", "_md5", "_hash", "_sign")
PIPELINE_META_FIELDS = frozenset({
    "_src_table", "_dst_table", "source_table",
    "source_id", "target_id", "join_field",
})
NOISE_NAME_KEYWORDS = (
    "remark", "detail", "template", "config",
)
CIPHERTEXT_MIN_LEN = 24
OVERSIZED_VALUE_LEN = 80
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


def load_entity_info(schema: Dict[str, Any]) -> Tuple[Dict[str, Tuple[str, str, str]], List[str]]:
    if not isinstance(schema.get("entities"), list):
        raise ValueError("The public schema entities must be a list")
    info: Dict[str, Tuple[str, str, str]] = {}
    for item in schema["entities"]:
        entity = canonical_entity(item["entity_type"])
        if not is_public_entity(entity):
            raise ValueError(f"Unknown public entity in schema: {entity}")
        info[entity] = (entity_category(entity), "", entity)
    return info, sorted(info.keys(), key=lambda x: -len(x))


def _rel_pair_key(fA: str, fB: str, entA: str, entB: str) -> str:
    return (f"{str(fA or '').strip()}&{str(fB or '').strip()}"
            f"|{str(entA or '').strip()}&{str(entB or '').strip()}")


def load_relation_resources(schema: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(schema.get("relations"), list):
        raise ValueError("The public schema relations must be a list")
    rel_by_key: Dict[str, str] = {}
    score_by_key: Dict[str, float] = {}
    for relation in schema["relations"]:
        label = relation["en"]
        rel_by_key[label] = label
        score_by_key[label] = TRADE_SCORE if relation["is_trade"] else 50.0
    return {"rel_by_key": rel_by_key, "score_by_key": score_by_key,
            "rel_by_pair": {}, "score_by_pair": {}}


def load_blacklist(path: str = BLACKLIST_PATH) -> Tuple[Set[str], Set[str], Set[str]]:
    both: Set[str] = set()
    attr: Set[str] = set()
    edge: Set[str] = set()
    if not os.path.isfile(path):
        return both, attr, edge
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if " @attr" in line:
                attr.add(line.replace("@attr", "").strip())
            elif " @edge" in line:
                edge.add(line.replace("@edge", "").strip())
            else:
                both.add(line)
    return both, attr, edge


def load_resources(schema_path: str = SCHEMA_PATH) -> Dict[str, Any]:
    with open(schema_path, "r", encoding="utf-8") as source:
        schema = json.load(source)
    if not isinstance(schema, dict) or not schema.get("entities"):
        raise ValueError("The graph schema must contain English entity metadata")
    entity_info, entity_names = load_entity_info(schema)
    rel = load_relation_resources(schema)
    field_dict = {"tables": {entity: sorted(public_entity_fields(entity))
                             for entity in entity_info}, "common": {}}
    bl_both, bl_attr, bl_edge = load_blacklist()
    return {
        "entity_info": entity_info,
        "entity_fields": {entity: public_entity_fields(entity) for entity in entity_info},
        "entity_names": entity_names,
        "rel_by_key": rel["rel_by_key"],
        "score_by_key": rel["score_by_key"],
        "rel_by_pair": rel["rel_by_pair"],
        "score_by_pair": rel["score_by_pair"],
        "field_dict": field_dict,
        "blacklist_attr": bl_both | bl_attr,
        "blacklist_edge": bl_both | bl_edge,
    }


def build_entity_matcher(entity_names: List[str]):
    names = sorted([n for n in entity_names if n], key=lambda x: -len(x))

    def match(vid: str) -> Optional[str]:
        if not vid:
            return None
        for n in names:
            if vid == n or vid.startswith(n + "_"):
                return n
        return None

    return match


def resolve_entity(entity: Optional[str], entity_info: Dict[str, Tuple[str, str, str]]):
    canonical = canonical_entity(entity or "")
    if canonical in entity_info:
        return entity_info[canonical]
    return ("", "", "")


def resolve_node_entity(node, matcher, entity_info):
    props = node.get("props") or {}
    for candidate in (node.get("entity"), node.get("vtype"), node.get("entity_type"),
                      props.get("vtype"), props.get("source_table"), props.get("entity_type")):
        entity = canonical_entity(candidate or "")
        info = resolve_entity(entity, entity_info)
        if any(info):
            return entity, info
    vid = node.get("vid") or node.get("unique_id") or ""
    entity = canonical_entity(matcher(vid) or "")
    return entity, resolve_entity(entity, entity_info)


def is_encrypted_field(field: str) -> bool:
    low = field.lower()
    return any(low.endswith(s) for s in ENCRYPTED_SUFFIXES)


def is_id_or_batch_field(field: str) -> bool:
    low = str(field).lower()
    if not low:
        return False
    if low == "id" or low.endswith("_id"):
        return True
    return "batch" in low


def is_noise_field_name(name: str) -> bool:
    if not name:
        return False
    low = str(name).lower()
    return any(kw in low for kw in NOISE_NAME_KEYWORDS)


def is_ciphertext_value(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    s = v.strip()
    if len(s) < CIPHERTEXT_MIN_LEN:
        return False
    if _BASE64_RE.match(s) and any(c in s for c in "+/="):
        return True
    if len(s) >= 40 and s.isalnum():
        has_lower = any(c.islower() for c in s)
        has_upper = any(c.isupper() for c in s)
        has_digit = any(c.isdigit() for c in s)
        if has_lower and has_upper and has_digit:
            return True
    return False


def is_oversized_value(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    s = v.strip()
    if len(s) > OVERSIZED_VALUE_LEN:
        return True

    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        return len(s) >= CIPHERTEXT_MIN_LEN
    return False


def is_empty_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def is_zero_timestamp(v: Any) -> bool:
    s = re.sub(r"\D", "", str(v))
    return len(s) >= 8 and set(s) == {"0"}


def format_timestamp(v: Any) -> Optional[Any]:
    s = str(v).strip()
    digits = re.sub(r"\D", "", s)
    if len(s) == 14 and len(digits) == 14:
        return f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]} {digits[8:10]}:{digits[10:12]}:{digits[12:14]}"
    if len(s) == 8 and len(digits) == 8:
        return f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}"
    return v


def parse_attrs(node: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(node.get("attributes"), dict):
        return dict(node["attributes"])
    props = node.get("props") or {}
    raw = props.get("attributes")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except (ValueError, TypeError):
            return {}
    fields = {field for entity in entity_definitions() for field in public_entity_fields(entity)}
    return {key: value for key, value in node.items() if key in fields}


def clean_node_attrs(
    node, matcher, res
) -> Tuple[Optional[str], Tuple[str, str, str], Dict[str, Any], Dict[str, Any]]:
    entity, (etype, esub, ecn) = resolve_node_entity(node, matcher, res["entity_info"])
    attrs = parse_attrs(node)
    blacklist = res["blacklist_attr"]
    allowed = (res.get("entity_fields") or {}).get(entity, public_entity_fields(entity))
    clean: Dict[str, Any] = {}
    clean_en: Dict[str, Any] = {}
    for field, value in attrs.items():
        if field not in allowed:
            continue
        if field in PIPELINE_META_FIELDS:
            continue
        if (field in blacklist or is_encrypted_field(field)
                or is_id_or_batch_field(field) or is_noise_field_name(field)):
            continue
        if is_empty_value(value) or is_zero_timestamp(value):
            continue
        if is_ciphertext_value(value) or is_oversized_value(value):
            continue
        formatted_value = format_timestamp(value)
        if is_empty_value(formatted_value):
            continue
        clean[field] = formatted_value

        clean_en[field] = value
    return entity, (etype, esub, ecn), clean, clean_en


def format_node(node, short_id, matcher, res) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    entity, (etype, esub, ecn), clean, clean_en = clean_node_attrs(node, matcher, res)
    out: Dict[str, Any] = {"id": short_id}
    if etype:
        out["category"] = etype
    if esub:
        out["entity_subtype"] = esub
    if entity:
        out["entity"] = entity
        out["entity_type"] = entity
    if ecn:
        out["entity_display_name"] = ecn
    if clean:
        out["attributes"] = clean
    return out, clean_en


def parse_join_field(join_field: str) -> Dict[str, Any]:
    out = {"kind": "shared", "left": "", "right": "", "role": "", "query": ""}
    value = str(join_field or "").strip()
    relation = public_relation(value)
    if relation:
        out["kind"] = "trade" if relation["is_trade"] else "shared"
        out["left"] = value
        out["query"] = value
        return out
    if not join_field or "|" not in join_field:
        out["left"] = join_field or ""
        out["query"] = (join_field or "").strip()
        return out
    left, right = join_field.split("|", 1)
    out["left"], out["right"] = left.strip(), right.strip()
    left_tokens = [t for t in left.split("&") if t]
    if len(left_tokens) == 1 and left_tokens[0].startswith("trade"):
        out["kind"] = "trade"
        out["role"] = right.strip().lower()
        out["query"] = left_tokens[0]
    else:
        out["kind"] = "shared"
        out["query"] = left_tokens[0] if left_tokens else ""
    return out


def _parsed_join_field(parsed: Dict[str, Any]) -> str:
    if parsed.get("right"):
        return f"{parsed.get('left', '')}|{parsed['right']}"
    return str(parsed.get("query") or "")


def is_relation_blacklisted(parsed: Dict[str, Any], res: Dict[str, Any]) -> bool:
    join_field = _parsed_join_field(parsed)
    if not public_relation(join_field):
        return True
    return parsed.get("query", "") in res.get("blacklist_edge", set())


def relation_label(parsed: Dict[str, Any], res: Dict[str, Any],
                   ent_pair: Optional[Tuple[str, str]] = None) -> str:
    relation = public_relation(_parsed_join_field(parsed))
    return str(relation.get("en") or "")


def relation_score(parsed: Dict[str, Any], res: Dict[str, Any],
                   ent_pair: Optional[Tuple[str, str]] = None) -> float:
    if parsed["kind"] == "trade":
        return TRADE_SCORE
    score_by_pair = res.get("score_by_pair") or {}
    if ent_pair and score_by_pair:
        lts = [t for t in parsed["left"].split("&") if t]
        fA = lts[0] if lts else parsed["query"]
        fB = lts[1] if len(lts) > 1 else ""
        pk = _rel_pair_key(fA, fB, ent_pair[0], ent_pair[1])
        if pk in score_by_pair:
            return float(score_by_pair[pk])
    return float(res["score_by_key"].get(parsed["query"], 50.0))


def aggregate_edges(edges: List[dict], res: Dict[str, Any],
                    vid2entity: Optional[Dict[str, str]] = None) -> List[dict]:
    agg: Dict[Tuple[str, str], dict] = {}
    order: List[Tuple[str, str]] = []
    for e in edges:
        src, dst = e.get("src", ""), e.get("dst", "")
        if not src or not dst:
            continue
        props = e.get("props") or e
        jf = canonical_relation_name(props.get("join_field") or props.get("relation") or "")
        parsed = parse_join_field(jf)
        if is_relation_blacklisted(parsed, res):
            continue
        key = (src, dst)
        if key not in agg:
            agg[key] = {"src": src, "dst": dst,
                        "first_hop": e.get("first_hop", 0), "relations": []}
            order.append(key)
        ent_pair = None
        if vid2entity:
            ent_pair = (vid2entity.get(src, ""), vid2entity.get(dst, ""))
        agg[key]["relations"].append({
            "label": relation_label(parsed, res, ent_pair=ent_pair),
            "score": relation_score(parsed, res, ent_pair=ent_pair),
            "amount": props.get("total_amount"),
            "trade_count": props.get("trade_count"),
            "top_k": props.get("top_k"),
            "raw_join_field": jf,
            "kind": parsed["kind"],
            "left": parsed["left"],
            "right": parsed["right"],
        })
    out = []
    for key in order:
        rec = agg[key]
        scores = [r["score"] for r in rec["relations"]]
        rec["importance_avg"] = round(sum(scores) / len(scores), 2) if scores else 0.0
        rec["relations"].sort(key=lambda r: -r["score"])
        out.append(rec)
    return out


def prune_nodes(nodes, agg_edges, target_id, avg_th=50.0, sum_th=150.0):
    node_edges: Dict[str, List[dict]] = defaultdict(list)
    for e in agg_edges:
        node_edges[e["src"]].append(e)
        node_edges[e["dst"]].append(e)
    drop: Set[str] = set()
    for n in nodes:
        vid = n.get("vid", "")
        if not vid or vid == target_id:
            continue
        conn = node_edges.get(vid, [])
        if not conn:
            drop.add(vid)
            continue
        scores = [e.get("importance_avg", 0.0) for e in conn]
        if (sum(scores) / len(scores)) < avg_th and sum(scores) < sum_th:
            drop.add(vid)
    kept_nodes = [n for n in nodes if n.get("vid", "") not in drop]
    kept_edges = [e for e in agg_edges if e["src"] not in drop and e["dst"] not in drop]
    return kept_nodes, kept_edges, drop


ENTITY_TYPE_PREFIX = {
    "account": "A",
    "object": "O",
    "operation": "B",
    "operation_behavior": "B",
    "subject": "S",
}


def assign_short_ids(nodes, target_id, matcher, res) -> Dict[str, str]:
    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for n in nodes:
        vid = n.get("vid", "")
        if not vid:
            continue

        entity, _ = resolve_node_entity(n, matcher, res["entity_info"])
        entity = entity or ""
        letter = entity_prefix(entity)
        grouped[(letter, entity)].append({
            "vid": vid,
            "hop": n.get("first_hop", 0),
            "is_target": vid == target_id,
        })
    group_meta: Dict[Tuple[str, str], dict] = {}
    for gkey, members in grouped.items():
        members.sort(key=lambda p: (0 if p["is_target"] else 1, p["hop"], p["vid"]))
        group_meta[gkey] = {
            "has_target": any(p["is_target"] for p in members),
            "min_hop": min(p["hop"] for p in members),
        }
    ordered_groups = sorted(
        grouped.keys(),
        key=lambda g: (
            0 if group_meta[g]["has_target"] else 1,
            group_meta[g]["min_hop"],
            g[0],
            g[1],
        ),
    )
    counters: Dict[str, int] = {}
    vid2short: Dict[str, str] = {}
    for gkey in ordered_groups:
        letter = gkey[0]
        for p in grouped[gkey]:
            vid = p["vid"]
            if vid in vid2short:
                continue
            counters[letter] = counters.get(letter, 0) + 1
            vid2short[vid] = f"{letter}{counters[letter]:03d}"
    return vid2short


def compress_short_ids(short_ids: List[str]) -> str:
    numeric: List[Tuple[str, int, str]] = []
    others: List[str] = []
    for sid in short_ids:
        m = re.match(r"^([A-Za-z]+)(\d+)$", sid or "")
        if m:
            numeric.append((m.group(1), int(m.group(2)), sid))
        elif sid:
            others.append(sid)
    numeric.sort(key=lambda p: (p[0], p[1]))
    segments: List[str] = []
    i = 0
    n = len(numeric)
    while i < n:
        j = i
        while (j + 1 < n and numeric[j + 1][0] == numeric[j][0]
               and numeric[j + 1][1] == numeric[j][1] + 1):
            j += 1
        if j == i:
            segments.append(numeric[i][2])
        else:
            segments.append(f"{numeric[i][2]}~{numeric[j][2]}")
        i = j + 1
    return ",".join(others + segments)


def short_id_sort_key(sid: str) -> Tuple[int, str, int]:
    m = re.match(r"^([A-Za-z]+)(\d+)$", sid or "")
    if m:
        return (0, m.group(1), int(m.group(2)))
    return (1, sid or "", 0)


def render_edge(agg_edge, vid2short, res) -> Dict[str, Any]:
    src = vid2short.get(agg_edge["src"], agg_edge["src"])
    dst = vid2short.get(agg_edge["dst"], agg_edge["dst"])
    items: List[str] = []
    for r in agg_edge["relations"]:
        label = r["raw_join_field"]
        if r["kind"] == "trade":
            parts = []
            if r.get("amount") is not None:
                parts.append(f"amount={r['amount']} fen")
            if r.get("trade_count") is not None:
                parts.append(f"count={r['trade_count']}")
            if r.get("top_k") is not None:
                parts.append(f"top_k={r['top_k']}")
            items.append(f"{label} ({', '.join(parts)})" if parts else label)
        else:
            items.append(label)
    return {f"{src}->{dst}": items}


def build_english_subgraph(item, kept_vids, node_en_attrs, res) -> Dict[str, Any]:
    matcher = build_entity_matcher(res["entity_names"])
    eng_nodes = []
    for n in item.get("nodes", []):
        vid = n.get("vid") or n.get("unique_id") or ""
        if vid not in kept_vids:
            continue
        entity, _ = resolve_node_entity(n, matcher, res["entity_info"])
        eng_nodes.append({
            "vid": vid,
            "entity": entity,
            "first_hop": n.get("first_hop", 0),
            "attributes": node_en_attrs.get(vid, {}),
        })
    eng_edges = []
    for e in item.get("edges", []):
        s, d = e.get("src", ""), e.get("dst", "")
        if s not in kept_vids or d not in kept_vids:
            continue
        props = e.get("props") or e
        join_field = canonical_relation_name(props.get("join_field") or props.get("relation") or "")
        if is_relation_blacklisted(parse_join_field(join_field), res):
            continue
        edge = {
            "src": s,
            "dst": d,
            "first_hop": e.get("first_hop", 0),
            "join_field": join_field,
        }
        if public_relation(join_field).get("is_trade"):
            for mk in ("total_amount", "trade_count", "top_k"):
                mv = props.get(mk)
                if not is_empty_value(mv):
                    edge[mk] = mv
        eng_edges.append(edge)
    src_stats = item.get("stats") or {}
    subgraph_stats = {
        "hops": item.get("hops"),
        "direction": item.get("direction"),
        "node_count": item.get("node_count"),
        "edge_count": item.get("edge_count"),
        "rel_set": src_stats.get("rel_set"),
        "distinct_relation_count": src_stats.get("distinct_relation_count"),
    }

    subgraph_stats = {k: v for k, v in subgraph_stats.items() if v is not None}
    return {"stats": subgraph_stats, "nodes": eng_nodes, "edges": eng_edges}


def format_subgraph(
    item: Dict[str, Any],
    res: Dict[str, Any],
    *,
    prune: bool = False,
    prune_avg_threshold: float = 50.0,
    prune_sum_threshold: float = 150.0,
) -> Dict[str, Any]:
    item = dict(item)
    nodes = [dict(node) for node in item.get("nodes", [])]
    for node in nodes:
        if not node.get("vid") and node.get("unique_id"):
            node["vid"] = node["unique_id"]
    target_id = str(item.get("target_id") or next(
        (node.get("vid") for node in nodes if int(node.get("first_hop", 0) or 0) == 0), "") or "")
    item["nodes"] = nodes
    edges = item.get("edges", [])
    matcher = build_entity_matcher(res["entity_names"])
    vid2entity: Dict[str, str] = {}
    for _n in nodes:
        _vid = _n.get("vid", "")
        if _vid:
            _ent, _ = resolve_node_entity(_n, matcher, res["entity_info"])
            if _ent:
                vid2entity[_vid] = _ent
    agg_edges = aggregate_edges(edges, res, vid2entity=vid2entity)
    if prune:
        nodes, agg_edges, _drop = prune_nodes(
            nodes, agg_edges, target_id, prune_avg_threshold, prune_sum_threshold)
    kept_vids = {n.get("vid", "") for n in nodes}
    vid2short = assign_short_ids(nodes, target_id, matcher, res)
    formatted: Dict[str, dict] = {}
    node_en_attrs: Dict[str, Dict[str, Any]] = {}
    for n in nodes:
        vid = n.get("vid", "")
        sid = vid2short.get(vid, vid)
        formatted[vid], node_en_attrs[vid] = format_node(n, sid, matcher, res)
    target_node = formatted.get(target_id)
    groups: Dict[Tuple[str, str, str, str], dict] = {}
    for n in nodes:
        f = formatted[n.get("vid", "")]
        key = (f.get("entity_type", ""), f.get("entity_subtype", ""),
               f.get("entity_display_name", ""), f.get("entity", ""))
        g = groups.get(key)
        if g is None:
            g = {"entity_type": key[0], "entity_subtype": key[1]}
            if key[2]:
                g["entity_display_name"] = key[2]
            if key[3]:
                g["entity"] = key[3]
            g["members"] = []
            groups[key] = g
        g["members"].append(f.get("id", ""))
    for g in groups.values():
        members = g["members"]
        g["count"] = len(members)
        g["members"] = compress_short_ids(members)
    tkey = ((target_node.get("entity_type", ""), target_node.get("entity_subtype", ""),
             target_node.get("entity_display_name", ""), target_node.get("entity", ""))
            if target_node else None)
    node_summary = [groups[k] for k in sorted(groups, key=lambda k: (0 if k == tkey else 1, k))]
    attr_groups: Dict[Tuple[str, str, str, str], dict] = {}
    for vid, f in formatted.items():
        if vid == target_id:
            continue
        attrs = f.get("attributes") or {}
        key = (f.get("entity_type", ""), f.get("entity_subtype", ""),
               f.get("entity_display_name", ""), f.get("entity", ""))
        g = attr_groups.get(key)
        if g is None:
            g = {"entity_type": key[0], "entity_subtype": key[1]}
            if key[2]:
                g["entity_display_name"] = key[2]
            if key[3]:
                g["entity"] = key[3]
            g["members"] = {}
            attr_groups[key] = g
        g["members"][f.get("id", "")] = attrs
    member_attrs = []
    for k in sorted(attr_groups, key=lambda k: (0 if k == tkey else 1, k)):
        g = attr_groups[k]
        g["members"] = {sid: g["members"][sid] for sid in sorted(g["members"])}
        member_attrs.append(g)
    agg_edges_sorted = sorted(
        agg_edges,
        key=lambda e: (short_id_sort_key(vid2short.get(e["src"], e["src"])),
                       short_id_sort_key(vid2short.get(e["dst"], e["dst"]))),
    )
    diff_edges = [render_edge(e, vid2short, res) for e in agg_edges_sorted]
    output: Dict[str, Any] = {"target_account": target_node, "node_summary": node_summary,
                              "relations": diff_edges, "member_attributes": member_attrs}
    output["english_subgraph"] = build_english_subgraph(item, kept_vids, node_en_attrs, res)
    return output
