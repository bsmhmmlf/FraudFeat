import json

import re

import time

from typing import Any, Dict, List, Optional, Tuple

from prompt_common import GREMLIN_OUTPUT_CONTRACT, MANUAL_FEATURE_SYSTEM_PROMPT, MANUAL_USER_PROMPT_TEMPLATE

from format_subgraph import load_resources, build_entity_matcher, resolve_entity, resolve_node_entity, parse_attrs, parse_join_field, relation_label, is_relation_blacklisted, is_encrypted_field, is_id_or_batch_field, is_noise_field_name, is_ciphertext_value, is_oversized_value, is_empty_value, is_zero_timestamp, PIPELINE_META_FIELDS, compress_short_ids, short_id_sort_key

from feature_utils import TARGET_SUBGRAPH_TOKENS, ENABLE_PRUNING, call_api, parse_model_response, estimate_tokens_simple, estimate_subgraph_tokens, subgraph_richness_stats, compute_vid_importance, rebuild_vid2short, _target_vid
from system.core.vocabulary import (canonical_entity, canonical_relation_name,
                                    entity_category,
                                    entity_definitions, entity_label, entity_prefix,
                                    public_entity_fields, public_relation)


PREFIX_LEGEND = {
    "A": "Account (account)",
    "O": "Object (object)",
    "B": "Operation behavior (operation_behavior)",
    "S": "Subject (subject)",
    "N": "Unrecognized",
}
FEATURE_SYSTEM_PROMPT_EN = MANUAL_FEATURE_SYSTEM_PROMPT
FEATURE_USER_PROMPT_TEMPLATE_EN = MANUAL_USER_PROMPT_TEMPLATE + GREMLIN_OUTPUT_CONTRACT


def normalize_input(raw: Dict[str, Any], res: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    matcher = build_entity_matcher(res["entity_names"])

    eng_field = raw.get("english_subgraph")
    if isinstance(eng_field, dict) and (eng_field.get("nodes") or eng_field.get("edges")):
        eng = eng_field
        target_vid = eng.get("target_vid") or _target_vid(eng)
        return eng, target_vid
    target_vid = raw.get("target_id") or ""
    nodes_out: List[Dict[str, Any]] = []
    for n in raw.get("nodes", []) or []:
        if not isinstance(n, dict):
            continue
        vid = n.get("vid") or n.get("unique_id") or ""
        entity, _info = resolve_node_entity(n, matcher, res["entity_info"])
        if entity not in entity_definitions():
            continue
        attrs = parse_attrs(n)
        nodes_out.append({
            "vid": vid,
            "entity": entity or "",
            "first_hop": int(n.get("first_hop", 0) or 0),
            "attributes": attrs,
        })
    known_vids = {node["vid"] for node in nodes_out}
    edges_out: List[Dict[str, Any]] = []
    for e in raw.get("edges", []) or []:
        if not isinstance(e, dict):
            continue
        props = e.get("props") or e
        join_field = canonical_relation_name(props.get("join_field") or props.get("relation") or "")
        public = public_relation(join_field)
        if not public or e.get("src") not in known_vids or e.get("dst") not in known_vids:
            continue
        edge = {"src": e.get("src", ""), "dst": e.get("dst", ""),
                "first_hop": int(e.get("first_hop", 0) or 0), "join_field": join_field}
        if public["is_trade"]:
            edge.update({"total_amount": props.get("total_amount"),
                         "trade_count": props.get("trade_count"),
                         "top_k": props.get("top_k")})
        edges_out.append(edge)
    if not target_vid:
        target_vid = _target_vid({"nodes": nodes_out})
    src_stats = raw.get("stats") or {}
    eng = {
        "stats": {
            "hops": raw.get("hops"),
            "direction": raw.get("direction"),
            "node_count": len(nodes_out),
            "edge_count": len(edges_out),
            "rel_set": src_stats.get("rel_set"),
            "distinct_relation_count": src_stats.get("distinct_relation_count"),
        },
        "nodes": nodes_out,
        "edges": edges_out,
    }
    return eng, target_vid


def clean_attributes(entity: Optional[str], attrs: Dict[str, Any],
                     res: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    blacklist = res["blacklist_attr"]
    allowed = (res.get("entity_fields") or {}).get(entity, public_entity_fields(entity or ""))
    out_en: Dict[str, Any] = {}
    field_names: Dict[str, str] = {}
    for field, value in (attrs or {}).items():
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
        out_en[field] = value
        field_names[field] = field
    return out_en, field_names


def build_mappings(eng: Dict[str, Any], res: Dict[str, Any]
                   ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, Any]],
                              Dict[str, str], Dict[str, Dict[str, str]],
                              Dict[str, Dict[str, str]]]:
    matcher = build_entity_matcher(res["entity_names"])
    entity_info = res["entity_info"]
    entity_meta: Dict[str, Dict[str, str]] = {}
    attr_clean: Dict[str, Dict[str, Any]] = {}
    attr_map: Dict[str, Dict[str, str]] = {}
    for n in eng.get("nodes", []) or []:
        if not isinstance(n, dict):
            continue
        vid = n.get("vid", "")
        entity = n.get("entity") or ""
        if not entity:
            entity = matcher(vid) or ""
        if entity in entity_info and entity not in entity_meta:
            etype, esub, display_name = resolve_entity(entity, entity_info)
            entity_meta[entity] = {
                "entity_type": entity,
                "category": etype,
                "entity_subtype": esub,
                "entity_display_name": display_name,
                "prefix": entity_prefix(entity),
            }
        attrs = n.get("attributes") or {}
        clean_en, field_names = clean_attributes(entity, attrs, res)
        attr_clean[vid] = clean_en
        if entity and field_names:
            am = attr_map.setdefault(entity, {})
            for field, name in field_names.items():
                am.setdefault(field, name)
    vid2entity = {n.get("vid", ""): n.get("entity", "")
                  for n in (eng.get("nodes") or []) if isinstance(n, dict)}
    rel_map: Dict[str, str] = {}
    for e in eng.get("edges", []) or []:
        if not isinstance(e, dict):
            continue
        jf = e.get("join_field", "")
        if not jf or jf in rel_map:
            continue
        parsed = parse_join_field(jf)
        if is_relation_blacklisted(parsed, res):
            continue
        ent_pair = (vid2entity.get(e.get("src", ""), ""), vid2entity.get(e.get("dst", ""), ""))
        label = relation_label(parsed, res, ent_pair=ent_pair)
        rel_map[jf] = label

        if parsed["kind"] == "shared" and "&" in parsed["left"] and "&" in parsed["right"]:
            lt = parsed["left"].split("&")
            rt = parsed["right"].split("&")
            mirror_jf = f"{lt[1].strip()}&{lt[0].strip()}|{rt[1].strip()}&{rt[0].strip()}"
            if mirror_jf not in rel_map:
                rel_map[mirror_jf] = label
    entity_map: Dict[str, Dict[str, str]] = {
        e: {"entity_type": e, "category": m["category"],
            "entity_subtype": m["entity_subtype"],
            "entity_display_name": m["entity_display_name"]}
        for e, m in entity_meta.items()
    }
    return entity_meta, attr_clean, rel_map, entity_map, attr_map


_INVALID_ENTITY_TOKENS = {"", "entity", "node", "unknown", "none", "null"}


def _looks_like_id_seg(seg: str) -> bool:
    if not seg:
        return False
    if seg.isdigit() or "@" in seg:
        return True
    if re.fullmatch(r"[0-9a-fA-F-]{16,}", seg):
        return True
    return False


def _resolve_entity_en(node: Dict[str, Any], entity_meta: Dict[str, Dict[str, str]]) -> str:
    ent = canonical_entity(node.get("entity") or node.get("entity_type") or "")
    if ent in entity_definitions():
        return ent
    if ent and ent.lower() not in _INVALID_ENTITY_TOKENS:
        return ""
    vid = node.get("vid", "") or ""
    return next((entity for entity in sorted(entity_definitions(), key=len, reverse=True)
                 if vid == entity or vid.startswith(entity + "_")), "")


def _ensure_entity_meta(entity: str, entity_meta: Dict[str, Dict[str, str]]) -> None:
    entity = canonical_entity(entity)
    if entity not in entity_definitions() or entity in entity_meta:
        return
    entity_meta[entity] = {
        "entity_type": entity,
        "category": entity_category(entity),
        "entity_subtype": "",
        "entity_display_name": entity_label(entity),
        "prefix": entity_prefix(entity),
    }


def rebuild_en_anon_subgraph(
    eng: Dict[str, Any],
    entity_meta: Dict[str, Dict[str, str]],
    attr_clean: Dict[str, Dict[str, Any]],
    kept_vids: set,
    target_vid: str,
    res: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    kept_nodes = [n for n in (eng.get("nodes") or []) if n.get("vid") in kept_vids]
    kept_edges_raw = [
        e for e in (eng.get("edges") or [])
        if isinstance(e, dict) and e.get("src") in kept_vids and e.get("dst") in kept_vids
    ]
    kept_nodes = [dict(n) for n in kept_nodes]
    for n in kept_nodes:
        ent = _resolve_entity_en(n, entity_meta)
        n["entity"] = ent
        _ensure_entity_meta(ent, entity_meta)
    kept_nodes = [n for n in kept_nodes if n.get("entity")]
    vid2short = rebuild_vid2short(kept_nodes, target_vid, entity_meta)
    tgt_node = next((n for n in kept_nodes if n.get("vid") == target_vid), None)
    target: Dict[str, Any] = {}
    tgt_entity = ""
    if tgt_node:
        tgt_entity = tgt_node.get("entity", "")
        target = {
            "id": vid2short.get(target_vid, target_vid),
            "entity": tgt_entity,
            "first_hop": tgt_node.get("first_hop", 0),
            "attributes": attr_clean.get(target_vid, {}),
        }
    groups: Dict[str, dict] = {}
    order_keys: List[str] = []
    tgt_key = tgt_entity
    for n in kept_nodes:
        vid = n.get("vid", "")
        entity = n.get("entity", "")
        g = groups.get(entity)
        if g is None:
            g = {"entity": entity, "_members": []}
            groups[entity] = g
            order_keys.append(entity)
        g["_members"].append(vid2short.get(vid, vid))

    def _group_sort_key(ent: str) -> tuple:
        meta = entity_meta.get(ent, {}) or {}
        return (0 if ent == tgt_key else 1, meta.get("prefix", "N"), ent)

    node_summary: List[dict] = []
    for ent in sorted(order_keys, key=_group_sort_key):
        g = groups[ent]
        members = g.pop("_members")
        g["members"] = compress_short_ids(members)
        g["count"] = len(members)
        node_summary.append(g)
    agg: Dict[Tuple[str, str], List[str]] = {}
    edge_order: List[Tuple[str, str]] = []
    for e in kept_edges_raw:
        parsed = parse_join_field(e.get("join_field", ""))
        if is_relation_blacklisted(parsed, res):
            continue
        if e.get("src") not in vid2short or e.get("dst") not in vid2short:
            continue
        s = vid2short[e.get("src", "")]
        d = vid2short[e.get("dst", "")]
        jf = e.get("join_field", "")
        if parsed["kind"] == "trade":
            parts = []
            amt = e.get("total_amount")
            cnt = e.get("trade_count")
            top = e.get("top_k")
            if amt is not None:
                parts.append(f"amount={amt} fen")
            if cnt is not None:
                parts.append(f"count={cnt}")
            if top is not None:
                parts.append(f"top_k={top}")
            label = f"{jf} ({', '.join(parts)})" if parts else jf
        else:
            label = jf
        k = (s, d)
        if k not in agg:
            agg[k] = []
            edge_order.append(k)
        if label and label not in agg[k]:
            agg[k].append(label)
    edge_order.sort(key=lambda k: (short_id_sort_key(k[0]), short_id_sort_key(k[1])))
    diff_edges = [{f"{k[0]}->{k[1]}": agg[k]} for k in edge_order]
    attr_groups: Dict[str, dict] = {}
    attr_order: List[str] = []
    for n in kept_nodes:
        vid = n.get("vid", "")
        if vid == target_vid:
            continue
        entity = n.get("entity", "")
        g = attr_groups.get(entity)
        if g is None:
            g = {"entity": entity, "members": {}}
            attr_groups[entity] = g
            attr_order.append(entity)
        sid = vid2short.get(vid, vid)
        g["members"][sid] = attr_clean.get(vid, {})
    member_attrs: List[dict] = []
    for ent in sorted(attr_order, key=_group_sort_key):
        g = attr_groups[ent]
        g["members"] = {sid: g["members"][sid] for sid in sorted(g["members"], key=short_id_sort_key)}
        member_attrs.append(g)
    subgraph_anon = {
        "target_account": target,
        "node_summary": node_summary,
        "relations": diff_edges,
        "member_attributes": member_attrs,
    }
    return subgraph_anon, vid2short


def prune_en_subgraph(
    eng: Dict[str, Any],
    entity_meta: Dict[str, Dict[str, str]],
    attr_clean: Dict[str, Dict[str, Any]],
    target_vid: str,
    res: Dict[str, Any],
    token_budget: int,
    rel_rarity: Dict[str, int] = None,
    rare_max: int = 10,
) -> Tuple[set, int, int]:
    nodes = eng.get("nodes", []) or []
    all_vids = [n.get("vid", "") for n in nodes if n.get("vid")]
    scores = compute_vid_importance(eng, target_vid, rel_rarity=rel_rarity, rare_max=rare_max)
    removable = sorted(
        (v for v in all_vids if scores.get(v, 0) != float("inf")),
        key=lambda v: scores.get(v, 0),
    )
    kept_vids = set(all_vids)

    def _tokens_for(kept: set) -> int:
        rebuilt, _ = rebuild_en_anon_subgraph(eng, entity_meta, attr_clean, kept, target_vid, res)
        return estimate_subgraph_tokens(rebuilt)

    cur_tokens = _tokens_for(kept_vids)
    if cur_tokens <= token_budget or not removable:
        return kept_vids, 0, cur_tokens
    pruned = 0
    idx = 0
    n_removable = len(removable)
    while idx < n_removable and cur_tokens > token_budget:
        over = cur_tokens - token_budget
        per_node = max(1, cur_tokens // max(1, len(kept_vids)))
        batch = max(1, min(over // per_node + 1, n_removable - idx))
        for _ in range(batch):
            if idx >= n_removable:
                break
            kept_vids.discard(removable[idx])
            idx += 1
            pruned += 1
        cur_tokens = _tokens_for(kept_vids)
    return kept_vids, pruned, cur_tokens


def prune_and_align_en_subgraph(
    eng: Dict[str, Any],
    target_vid: str,
    res: Dict[str, Any],
    entity_meta: Dict[str, Dict[str, str]],
    attr_clean: Dict[str, Dict[str, Any]],
    mapping_table: Dict[str, Any],
    token_budget: int = TARGET_SUBGRAPH_TOKENS,
    rel_rarity: Dict[str, int] = None,
    rare_max: int = 10,
) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, Any], Dict[str, Any]]:
    orig_nodes = len(eng.get("nodes") or [])
    mapping_tokens = estimate_tokens_simple(json.dumps(mapping_table, ensure_ascii=False, indent=2))
    subgraph_budget = max(8_000, token_budget - mapping_tokens)
    orig_subgraph_anon, _ = rebuild_en_anon_subgraph(
        eng, entity_meta, attr_clean, {n.get("vid") for n in (eng.get("nodes") or [])},
        target_vid, res)
    orig_tokens = estimate_subgraph_tokens(orig_subgraph_anon)
    if not ENABLE_PRUNING or orig_tokens <= subgraph_budget:
        kept_vids = {n.get("vid", "") for n in (eng.get("nodes") or []) if n.get("vid")}
        subgraph_anon, vid2short = rebuild_en_anon_subgraph(
            eng, entity_meta, attr_clean, kept_vids, target_vid, res)
        eng_pruned = _filter_eng(eng, kept_vids)
        return subgraph_anon, vid2short, eng_pruned, {
            "orig_tokens": orig_tokens, "final_tokens": orig_tokens,
            "mapping_tokens": mapping_tokens, "subgraph_budget": subgraph_budget,
            "orig_nodes": orig_nodes, "kept_nodes": orig_nodes, "pruned": 0,
        }
    kept_vids, pruned, final_tokens = prune_en_subgraph(
        eng, entity_meta, attr_clean, target_vid, res, subgraph_budget,
        rel_rarity=rel_rarity, rare_max=rare_max)
    subgraph_anon, vid2short = rebuild_en_anon_subgraph(
        eng, entity_meta, attr_clean, kept_vids, target_vid, res)
    eng_pruned = _filter_eng(eng, kept_vids)
    return subgraph_anon, vid2short, eng_pruned, {
        "orig_tokens": orig_tokens, "final_tokens": final_tokens,
        "mapping_tokens": mapping_tokens, "subgraph_budget": subgraph_budget,
        "orig_nodes": orig_nodes, "kept_nodes": len(kept_vids), "pruned": pruned,
    }


def _filter_eng(eng: Dict[str, Any], kept_vids: set) -> Dict[str, Any]:
    kept_nodes = [n for n in (eng.get("nodes") or []) if n.get("vid") in kept_vids]
    kept_edges = [
        e for e in (eng.get("edges") or [])
        if isinstance(e, dict) and e.get("src") in kept_vids and e.get("dst") in kept_vids
    ]
    stats = dict(eng.get("stats") or {})
    stats["node_count"] = len(kept_nodes)
    stats["edge_count"] = len(kept_edges)
    rel_set = sorted({e.get("join_field", "") for e in kept_edges
                      if isinstance(e, dict) and e.get("join_field")})
    if "rel_set" in stats or rel_set:
        stats["rel_set"] = rel_set
        stats["distinct_relation_count"] = len(rel_set)
    hops = [int(n.get("first_hop", 0) or 0) for n in kept_nodes]
    hops += [int(e.get("first_hop", 0) or 0) for e in kept_edges]
    if hops:
        stats["hops"] = max(hops)
    return {"stats": stats, "nodes": kept_nodes, "edges": kept_edges}


def process_subgraph_en(
    subgraph_raw: Dict[str, Any],
    index: int,
    model: str,
    res: Dict[str, Any],
    temperature: float = 0.9,
    rel_rarity: Dict[str, int] = None,
    rare_max: int = 10,
) -> Dict[str, Any]:
    eng, target_vid = normalize_input(subgraph_raw, res)
    if not target_vid:
        target_vid = _target_vid(eng)
    tgt_node = next((n for n in (eng.get("nodes") or []) if n.get("vid") == target_vid), None)
    entity_name = (tgt_node or {}).get("entity", "unknown")
    print(f"  [{index}] Target: {target_vid} ({entity_name})")
    has_nodes = len(eng.get("nodes") or []) > 1
    has_edges = len(eng.get("edges") or []) > 0
    if not (has_nodes or has_edges):
        print("    [SKIP] No neighborhood expansion")
        return {
            "index": index, "target_id": target_vid, "entity_name": entity_name,
            "status": "skipped", "reason": "subgraph contains only the target node",
        }
    entity_meta, attr_clean, rel_map, entity_map, attr_map = build_mappings(eng, res)
    mapping_table_for_llm: Dict[str, Any] = {
        "entity_mapping": entity_map,
        "relation_mapping": rel_map,
        "attribute_mapping": attr_map,
        "prefix_legend": dict(PREFIX_LEGEND),
    }
    subgraph_anon, vid2short, eng_pruned, prune_info = prune_and_align_en_subgraph(
        eng, target_vid, res, entity_meta, attr_clean, mapping_table_for_llm,
        token_budget=TARGET_SUBGRAPH_TOKENS,
        rel_rarity=rel_rarity, rare_max=rare_max)
    if prune_info["pruned"] > 0:
        print(f"    Pruned nodes: {prune_info['orig_nodes']} -> {prune_info['kept_nodes']}; "
              f"subgraph tokens {prune_info['orig_tokens']} -> {prune_info['final_tokens']} "
              f"(mapping tokens~{prune_info['mapping_tokens']}, subgraph budget {prune_info['subgraph_budget']})")
    else:
        print(f"    Subgraph tokens~{prune_info['final_tokens']} + mapping tokens~{prune_info['mapping_tokens']}"
              f" <= total budget {TARGET_SUBGRAPH_TOKENS}; no pruning needed")
    max_hop = max(1, int(subgraph_richness_stats(
        {"english_subgraph": eng_pruned}).get("max_hop", 0) or 0))
    if prune_info.get("pruned", 0) > 0:
        orig_max_hop = int(subgraph_richness_stats(
            {"english_subgraph": eng}).get("max_hop", 0) or 0)
        flag = "depth preserved" if max_hop >= orig_max_hop else "[WARN] depth reduced"
        print(f"    max_hop before pruning={orig_max_hop} -> after pruning={max_hop}; {flag}")
    else:
        print(f"    Actual maximum subgraph depth: max_hop={max_hop}")
    from prompt_common import prompt_data
    llm_input = dict(subgraph_anon)
    llm_input["mappings"] = mapping_table_for_llm
    subgraph_json = json.dumps(prompt_data(llm_input), ensure_ascii=False, indent=2)
    from string import Template
    user_prompt = Template(FEATURE_USER_PROMPT_TEMPLATE_EN).safe_substitute(
        target_id=target_vid, subgraph_json=subgraph_json, max_hop=max_hop)
    pre_tokens = estimate_subgraph_tokens(eng)
    llm_body_tokens = estimate_subgraph_tokens(llm_input)
    full_prompt_tokens = estimate_tokens_simple(
        FEATURE_SYSTEM_PROMPT_EN + user_prompt)
    print(f"    Token estimate: original subgraph~{pre_tokens:,} -> full prompt~{full_prompt_tokens:,} "
          f"(subgraph + mappings~{llm_body_tokens:,})")
    token_estimate = {
        "pre_tokens": pre_tokens,
        "llm_body_tokens": llm_body_tokens,
        "full_prompt_tokens": full_prompt_tokens,
    }
    print(f"    Calling API ({model})...")
    t0 = time.time()
    response = call_api(FEATURE_SYSTEM_PROMPT_EN, user_prompt, model=model, temperature=temperature)
    elapsed = time.time() - t0
    if response is None:
        return {
            "index": index, "target_id": target_vid, "entity_name": entity_name,
            "status": "error", "reason": "API request failed",
        }
    print(f"    Completed ({elapsed:.1f}s, {len(response)} chars)")
    parsed = parse_model_response(response)
    if isinstance(parsed, dict):
        _all_hops = []
        for _rp in (parsed.get("fraud_patterns") or parsed.get("risk_patterns") or []):
            for _f in (_rp.get("features") or []):
                try:
                    _all_hops.append(int(_f.get("hops", 0)))
                except (ValueError, TypeError):
                    pass
        if _all_hops:
            _hit = sum(1 for h in _all_hops if h == max_hop)
            print(f"    Feature hops={_all_hops} | reaching max_hop={max_hop}: {_hit}/{len(_all_hops)}")
    anon_mapping_full = {sid: vid for vid, sid in vid2short.items()}

    mapping_table_full = dict(mapping_table_for_llm)
    mapping_table_full["anonymization_mapping"] = anon_mapping_full
    result = {
        "index": index,
        "target_id": target_vid,
        "entity_name": entity_name,
        "status": "success",
        "model": model,
        "temperature": temperature,
        "elapsed_seconds": round(elapsed, 1),
        "prune_info": prune_info,
        "token_estimate": token_estimate,
        "analysis": parsed if parsed is not None else response,
        "mappings": mapping_table_full,
        "anonymized_english_subgraph": subgraph_anon,
        "original_english_subgraph": eng,
    }
    if parsed is None:
        result["parse_error"] = True
        print("    [WARN] Model output is not valid JSON; raw text was retained in analysis")
    return result
