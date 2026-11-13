import json

import time

import math

import logging

import requests

import re

import copy

import os

from collections import defaultdict

from typing import Dict, List, Any, Tuple, Optional, Set

from datetime import datetime

from prompt_common import CONTEXTUAL_FEATURE_SYSTEM_PROMPT, build_stat_round1_prompt as build_shared_round1_prompt, build_stat_round2_prompt as build_shared_round2_prompt, build_stat_final_user_prompt as build_shared_final_user_prompt


API_URL = os.environ.get("SYSTEM_LLM_API_URL", "<YOUR_LLM_API_URL>")
API_KEY = os.environ.get("SYSTEM_LLM_API_KEY", "<YOUR_LLM_API_KEY>")
MODEL_NAME = "hy4-preview"

API_TIMEOUT = 600

MAX_OUTPUT_TOKENS = 8000

VERBOSE_LLM = True

ENABLE_PRUNING = True

FIELD_THRESHOLD = 60

FIELD_IMPORTANCE = {}

RELATION_IMPORTANCE = {}

BLACKLIST_TYPES = set()

TARGET_SUBGRAPH_TOKENS = 96000

logger = logging.getLogger(__name__)


def estimate_tokens_simple(text: str) -> int:
    chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    others = len(text) - chinese
    return int(chinese * 2 + others * 0.25)


def estimate_subgraph_tokens(subgraph: Dict[str, Any]) -> int:
    subgraph_json = json.dumps(subgraph, ensure_ascii=False, indent=2)
    return estimate_tokens_simple(subgraph_json)


_MEMBER_RANGE_RE = re.compile(r"^([A-Za-z]+)(\d+)~([A-Za-z]+)(\d+)$")
_SHORT_ID_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def _expand_short_ids(members: Any) -> List[str]:
    if isinstance(members, dict):
        return [str(k) for k in members]
    if isinstance(members, str):
        members = [seg for seg in members.split(",") if seg.strip()]
    if not isinstance(members, (list, tuple)):
        return []
    out: List[str] = []
    for m in members:
        if isinstance(m, dict):
            nid = m.get("id", "")
            if nid:
                out.append(str(nid))
            continue
        s = str(m).strip()
        if not s:
            continue
        mm = _MEMBER_RANGE_RE.match(s)
        if mm and mm.group(1) == mm.group(3):
            width = len(mm.group(2))
            out.extend(f"{mm.group(1)}{i:0{width}d}"
                       for i in range(int(mm.group(2)), int(mm.group(4)) + 1))
        else:
            out.append(s)
    return out


def _compress_short_ids(ids: List[str]) -> Any:
    parsed: List[Tuple[str, int, str]] = []
    for sid in ids:
        mm = _SHORT_ID_RE.match(str(sid))
        if not mm:
            return list(ids)
        parsed.append((mm.group(1), int(mm.group(2)), str(sid)))
    if not parsed:
        return list(ids)
    parsed.sort(key=lambda p: (p[0], p[1]))
    cont = all(parsed[i + 1][0] == parsed[i][0]
               and parsed[i + 1][1] == parsed[i][1] + 1
               for i in range(len(parsed) - 1))
    if cont:
        return parsed[0][2] if len(parsed) == 1 else f"{parsed[0][2]}~{parsed[-1][2]}"
    return list(ids)


def _is_en_anon_schema(subgraph: Dict[str, Any]) -> bool:
    if not isinstance(subgraph, dict):
        return False
    return (any(isinstance(group, dict) and "members" in group
                for group in (subgraph.get("node_summary") or []))
            or any(isinstance(group, dict) and isinstance(group.get("members"), dict)
                   for group in (subgraph.get("member_attributes") or [])))


def _en_anon_view(subgraph: Dict[str, Any]) -> Dict[str, Any]:
    if not _is_en_anon_schema(subgraph):
        return subgraph
    target = dict(subgraph.get("target_account") or {})
    summary = [{"id": sid, "entity": g.get("entity", ""),
                "entity_type": g.get("entity_type", "")}
               for g in (subgraph.get("node_summary") or [])
               for sid in _expand_short_ids(g.get("members"))]
    members = [{"id": sid, "entity": g.get("entity", ""),
                "attributes": attrs}
               for g in (subgraph.get("member_attributes") or [])
               for sid, attrs in (g.get("members") or {}).items()]
    rels: List[Dict[str, Any]] = []
    for rel in (subgraph.get("relations") or []):
        if not isinstance(rel, dict):
            continue
        if "src" in rel:
            rels.append(rel)
            continue
        for key, labels in rel.items():
            ks = str(key)
            if "->" not in ks:
                continue
            s, d = ks.split("->", 1)
            rels.append({"src": s.strip(), "dst": d.strip(),
                         "join_fields": [str(x) for x in (labels or [])]})
    view = dict(subgraph)
    view["target_account"] = target
    view["node_summary"] = summary
    view["member_attributes"] = members
    view["relations"] = rels
    return view


def _keep_by_ids(subgraph: Dict[str, Any], node_ids, *, keep: bool) -> Dict[str, Any]:
    sel = set(str(x) for x in (node_ids or []))

    def _hit(nid: str) -> bool:
        return (str(nid) in sel) if keep else (str(nid) not in sel)

    pruned = copy.deepcopy(subgraph)
    if "relation_groups" in pruned:
        new_groups = []
        for group in (pruned.get("relation_groups") or []):
            members = []
            for member in (group.get("members") or []):
                nid = member.get("id", "") if isinstance(member, dict) else str(member)
                if nid and _hit(nid):
                    members.append(member)
            if members:
                g = dict(group)
                g["members"] = members
                new_groups.append(g)
        pruned["relation_groups"] = new_groups

    if "node_summary" in pruned:
        new_summary = []
        for node in (pruned.get("node_summary") or []):
            if "members" in node and not isinstance(node.get("members"), dict):
                ids = [s for s in _expand_short_ids(node.get("members")) if _hit(s)]
                if ids:
                    nd = dict(node)
                    nd["members"] = _compress_short_ids(ids)
                    nd["count"] = len(ids)
                    new_summary.append(nd)
            else:
                nid = node.get("id", "")
                if nid and _hit(nid):
                    new_summary.append(node)
        pruned["node_summary"] = new_summary
    if "member_attributes" in pruned:
        new_members = []
        for member in (pruned.get("member_attributes") or []):
            if isinstance(member.get("members"), dict):
                kept = {sid: attrs for sid, attrs in member["members"].items()
                        if _hit(sid)}
                if kept:
                    m = dict(member)
                    m["members"] = kept
                    new_members.append(m)
            else:
                nid = member.get("id", "")
                if nid and _hit(nid):
                    new_members.append(member)
        pruned["member_attributes"] = new_members
    if "relations" in pruned:
        new_rels = []
        for rel in (pruned.get("relations") or []):
            if not isinstance(rel, dict):
                continue
            if "src" in rel:
                s, d = str(rel.get("src", "")), str(rel.get("dst", ""))
                if s and d and _hit(s) and _hit(d):
                    new_rels.append(rel)
                continue
            for key, labels in rel.items():
                ks = str(key)
                if "->" not in ks:
                    continue
                s, d = (x.strip() for x in ks.split("->", 1))
                if s and d and _hit(s) and _hit(d):
                    new_rels.append({key: labels})
        pruned["relations"] = new_rels
    return pruned


def get_field_score(field_name: str) -> Optional[float]:
    global FIELD_IMPORTANCE
    if field_name in FIELD_IMPORTANCE:
        return FIELD_IMPORTANCE[field_name]
    field_lower = field_name.lower()
    for key, score in FIELD_IMPORTANCE.items():
        if key.lower() == field_lower:
            return score
    for key, score in FIELD_IMPORTANCE.items():
        key_lower = key.lower()
        if (key_lower in field_lower or field_lower in key_lower) and \
           len(key_lower) >= 3 and len(field_lower) >= 3:
            return score
    return None


def should_keep_field(field_name: str, threshold: int = FIELD_THRESHOLD) -> bool:
    score = get_field_score(field_name)
    if score is None:
        return True
    return score >= threshold


def prune_fields_from_subgraph(subgraph: Dict[str, Any]) -> Dict[str, Any]:
    pruned = copy.deepcopy(subgraph)
    new_members = []
    for member in pruned.get("member_attributes", []):
        new_member = {}
        for key, value in member.items():
            if should_keep_field(key):
                new_member[key] = value
        new_members.append(new_member)

    pruned["member_attributes"] = new_members
    return pruned


def extract_field_en(field_list: List[str]) -> List[str]:
    result = []
    for field in (field_list or []):
        text = str(field).strip()
        match = re.search(r'\(([A-Za-z0-9_&|]+)\)', text)
        identifier = match.group(1) if match else text.split(" (", 1)[0]
        if identifier:
            result.append(identifier)
    return result


def get_rel_score(src_id: str, tgt_id: str,
                  src_type: str, tgt_type: str,
                  field_en_list: List[str],
                  rel_score_cache: Dict) -> float:
    TX_FIELDS = {"total_amount", "trade_count"}
    best_score = 0
    for field in field_en_list:
        key = (src_type, tgt_type, field)
        if key in rel_score_cache:
            best_score = max(best_score, rel_score_cache[key])
        if field in TX_FIELDS:
            best_score = max(best_score, 100)
    if best_score == 0:
        key = (src_type, tgt_type, "*")
        best_score = rel_score_cache.get(key, 15)
    return best_score


def is_tx_rel(field_en_list: List[str]) -> bool:
    TX_KEYWORDS = {"total_amount", "trade_count", "payer", "payee",
                   "withdraw", "charge", "from_card"}
    for f in field_en_list:
        if any(kw in f for kw in TX_KEYWORDS):
            return True
    return False


def compute_node_importance(subgraph: Dict[str, Any], rel_score_cache: Dict) -> Dict[str, float]:
    subgraph = _en_anon_view(subgraph)
    importance = {}
    target_id = get_target_id(subgraph)
    node_type_en = {}
    all_node_ids = set()
    for node in subgraph.get("node_summary", []):
        nid = node.get("id", "")
        if nid:
            node_type_en[nid] = ""
            all_node_ids.add(nid)
    for group in subgraph.get("relation_groups", []):
        for member_id in group.get("members", []):
            if isinstance(member_id, dict):
                nid = member_id.get("id", "")
            else:
                nid = str(member_id)
            if nid:
                all_node_ids.add(nid)
                if nid not in node_type_en:
                    node_type_en[nid] = ""
    for member in subgraph.get("member_attributes", []):
        nid = member.get("id", "")
        if nid:
            all_node_ids.add(nid)
            entity_name_en = member.get("entity", "")
            if entity_name_en:
                node_type_en[nid] = entity_name_en

    if target_id:
        importance[target_id] = float('inf')
    black_nodes = set()
    for nid, entity_name_en in node_type_en.items():
        if entity_name_en in BLACKLIST_TYPES:
            importance[nid] = float('inf')
            black_nodes.add(nid)
    for rel in subgraph.get("relations", []):
        src = rel.get("src", "")
        tgt = rel.get("dst", "")
        if src in black_nodes and tgt:
            importance[tgt] = float('inf')
        if tgt in black_nodes and src:
            importance[src] = float('inf')
    pair_data = {}
    for rel in subgraph.get("relations", []):
        src = rel.get("src", "")
        tgt = rel.get("dst", "")
        if not src or not tgt:
            continue
        pair_key = tuple(sorted((src, tgt)))
        if pair_key not in pair_data:
            pair_data[pair_key] = {"scores": [], "has_tx": False}
        src_type_en = node_type_en.get(src, "")
        tgt_type_en = node_type_en.get(tgt, "")
        fields = extract_field_en(rel.get("join_fields", []))
        if is_tx_rel(fields):
            pair_data[pair_key]["has_tx"] = True
        score = get_rel_score(src, tgt, src_type_en, tgt_type_en, fields, rel_score_cache)
        pair_data[pair_key]["scores"].append(score)

    node_pair_scores = {}
    node_degree = {}
    for (n1, n2), data in pair_data.items():
        if data["has_tx"]:
            final_pair_score = 100
        else:
            final_pair_score = sum(data["scores"])
        for nid in (n1, n2):
            if nid not in node_pair_scores:
                node_pair_scores[nid] = []
            node_pair_scores[nid].append(final_pair_score)
            node_degree[nid] = node_degree.get(nid, 0) + 1
    for node_id, scores in node_pair_scores.items():
        if node_id in importance:
            continue
        sum_score = sum(scores)
        deg = node_degree.get(node_id, 1)
        final_score = sum_score / math.log(deg + 1)
        importance[node_id] = final_score
    for nid in all_node_ids:
        if nid not in importance:
            importance[nid] = 10.0
    return importance


def remove_nodes_from_subgraph(subgraph: Dict[str, Any], node_ids: List[str]) -> Dict[str, Any]:
    return _keep_by_ids(subgraph, node_ids, keep=False)


def prune_subgraph(subgraph: Dict[str, Any],
                   rel_score_cache: Dict = None) -> Dict[str, Any]:
    global ENABLE_PRUNING, TARGET_SUBGRAPH_TOKENS, RELATION_IMPORTANCE

    if not ENABLE_PRUNING:
        return subgraph
    if rel_score_cache is None:
        rel_score_cache = RELATION_IMPORTANCE

    original_tokens = estimate_subgraph_tokens(subgraph)
    if original_tokens <= TARGET_SUBGRAPH_TOKENS:
        return subgraph
    pruned = prune_fields_from_subgraph(subgraph)
    pruned_tokens = estimate_subgraph_tokens(pruned)
    if pruned_tokens <= TARGET_SUBGRAPH_TOKENS:
        return pruned

    importance = compute_node_importance(pruned, rel_score_cache)
    immutable_nodes = []
    mutable_nodes = []
    for node_id, score in importance.items():
        if score == float('inf'):
            immutable_nodes.append(node_id)
        else:
            mutable_nodes.append((node_id, score))
    mutable_nodes.sort(key=lambda x: x[1])
    buckets = defaultdict(list)
    for node_id, score in mutable_nodes:
        bucket_key = int(score // 10) * 10
        buckets[bucket_key].append((node_id, score))
    sorted_buckets = sorted(buckets.keys())
    current_subgraph = copy.deepcopy(pruned)
    for bucket_key in sorted_buckets:
        bucket_nodes = buckets[bucket_key]
        bucket_node_ids = [nid for nid, _ in bucket_nodes]
        test_subgraph = remove_nodes_from_subgraph(current_subgraph, bucket_node_ids)
        test_tokens = estimate_subgraph_tokens(test_subgraph)
        if test_tokens > TARGET_SUBGRAPH_TOKENS:
            current_subgraph = test_subgraph
            continue
        else:
            for node_id, score in bucket_nodes:
                test_subgraph = remove_nodes_from_subgraph(current_subgraph, [node_id])
                test_tokens = estimate_subgraph_tokens(test_subgraph)

                if test_tokens > TARGET_SUBGRAPH_TOKENS:
                    current_subgraph = test_subgraph
                    continue
                else:
                    current_subgraph = test_subgraph
                    break

            break
    return current_subgraph


def get_target_id(subgraph: Dict[str, Any]) -> str:
    target = subgraph.get("target_account", {})
    return target.get("id", "unknown")


def get_target_entity_name(subgraph: Dict[str, Any]) -> str:
    target = subgraph.get("target_account", {})
    return target.get("entity") or target.get("entity_display_name") or "unknown"


def _collect_node_ids(subgraph: Dict[str, Any]) -> List[str]:
    subgraph = _en_anon_view(subgraph)
    node_ids: List[str] = []

    target = subgraph.get("target_account", {})
    if isinstance(target, dict):
        nid = target.get("id", "")
        if nid:
            node_ids.append(nid)

    for node in subgraph.get("node_summary", []):
        nid = node.get("id", "")
        if nid:
            node_ids.append(nid)

    for group in subgraph.get("relation_groups", []):
        for member in group.get("members", []):
            if isinstance(member, dict):
                nid = member.get("id", "")
            else:
                nid = str(member)
            if nid:
                node_ids.append(nid)
    for member in subgraph.get("member_attributes", []):
        nid = member.get("id", "")
        if nid:
            node_ids.append(nid)
    return node_ids


def _filter_original_by_kept_ids(original: Dict[str, Any], kept_ids: List[str]) -> Dict[str, Any]:
    kept_set = set(kept_ids)
    orig_nodes = original.get("nodes", [])
    orig_edges = original.get("edges", [])
    filtered_nodes = [n for n in orig_nodes if n.get("vid", "") in kept_set]
    filtered_edges = [
        e for e in orig_edges
        if e.get("src", "") in kept_set and e.get("dst", "") in kept_set
    ]
    return {
        "nodes": filtered_nodes,
        "edges": filtered_edges,
    }


def is_trivial_subgraph(subgraph: Dict[str, Any]) -> bool:
    has_groups = len(subgraph.get("relation_groups", [])) > 0
    has_members = len(subgraph.get("member_attributes", [])) > 0
    has_diff_rels = len(subgraph.get("relations", [])) > 0
    if has_groups or has_members or has_diff_rels:
        return False
    return True


def _prettify_json_response(text: str) -> str:
    if not isinstance(text, str):
        return text
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        parsed = json.loads(cleaned)
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, ValueError):
        return text


def _build_adjacency_from_subgraph(subgraph: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, int]], Set[str]]:
    subgraph = _en_anon_view(subgraph)
    adjacency: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    all_nodes: Set[str] = set()

    target = subgraph.get("target_account", {})
    if isinstance(target, dict):
        target_id = target.get("id", "")
        if target_id:
            all_nodes.add(target_id)
    for group in subgraph.get("relation_groups", []):
        members = group.get("members", [])
        common_fields = group.get("shared_fields", [])
        field_count = len(common_fields)
        member_ids = []
        for member in members:
            if isinstance(member, dict):
                nid = member.get("id", "")
            else:
                nid = str(member)
            if nid:
                member_ids.append(nid)
                all_nodes.add(nid)
        for i in range(len(member_ids)):
            for j in range(i + 1, len(member_ids)):
                a, b = member_ids[i], member_ids[j]
                adjacency[a][b] += field_count
                adjacency[b][a] += field_count
    for rel in subgraph.get("relations", []):
        src = rel.get("src", "")
        tgt = rel.get("dst", "")
        fields = rel.get("join_fields", [])
        field_count = len(fields)

        if src and tgt and field_count > 0:
            all_nodes.add(src)
            all_nodes.add(tgt)
            adjacency[src][tgt] += field_count
            adjacency[tgt][src] += field_count
    for nid in all_nodes:
        if nid not in adjacency:
            adjacency[nid] = defaultdict(int)
    return dict(adjacency), all_nodes


def _compute_degrees(adjacency: Dict[str, Dict[str, int]], all_nodes: Set[str]) -> Dict[str, int]:
    degrees = {}
    for node in all_nodes:
        neighbors = adjacency.get(node, {})
        degrees[node] = sum(neighbors.values())
    return degrees


def _compute_clustering_coefficient(adjacency: Dict[str, Dict[str, int]], node: str) -> float:
    neighbors = set(adjacency.get(node, {}).keys())
    k = len(neighbors)
    if k < 2:
        return 0.0
    actual_edges = 0
    neighbor_list = list(neighbors)
    for i in range(len(neighbor_list)):
        for j in range(i + 1, len(neighbor_list)):
            ni, nj = neighbor_list[i], neighbor_list[j]
            if ni in adjacency and nj in adjacency[ni] and adjacency[ni][nj] > 0:
                actual_edges += 1
    max_edges = k * (k - 1) / 2
    return actual_edges / max_edges


def _compute_betweenness_centrality(adjacency: Dict[str, Dict[str, int]], all_nodes: Set[str]) -> Dict[str, float]:
    betweenness = {node: 0.0 for node in all_nodes}
    nodes_list = list(all_nodes)
    for s in nodes_list:
        stack = []
        predecessors = {node: [] for node in all_nodes}
        sigma = {node: 0 for node in all_nodes}
        sigma[s] = 1
        dist = {node: -1 for node in all_nodes}
        dist[s] = 0
        queue = [s]
        qi = 0
        while qi < len(queue):
            v = queue[qi]
            qi += 1
            stack.append(v)
            for w in adjacency.get(v, {}).keys():
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)
        delta = {node: 0.0 for node in all_nodes}
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                betweenness[w] += delta[w]
    n = len(all_nodes)
    if n > 2:
        norm = (n - 1) * (n - 2)
        for node in all_nodes:
            betweenness[node] = betweenness[node] / norm if norm > 0 else 0.0
    else:
        for node in all_nodes:
            betweenness[node] = 0.0
    return betweenness


def _compute_kcore(adjacency: Dict[str, Dict[str, int]], all_nodes: Set[str]) -> Dict[str, int]:
    simple_degree = {}
    for node in all_nodes:
        simple_degree[node] = len(adjacency.get(node, {}))

    core_number = {}
    remaining = set(all_nodes)
    current_k = 0
    while remaining:
        min_deg = min(simple_degree[n] for n in remaining)
        current_k = max(current_k, min_deg)

        to_remove = set()
        changed = True
        while changed:
            changed = False
            for node in remaining:
                if node in to_remove:
                    continue
                if simple_degree[node] <= current_k:
                    to_remove.add(node)
                    core_number[node] = current_k
                    for neighbor in adjacency.get(node, {}).keys():
                        if neighbor in remaining and neighbor not in to_remove:
                            simple_degree[neighbor] -= 1
                    changed = True
        remaining -= to_remove
    return core_number


def compute_graph_statistics(subgraph: Dict[str, Any]) -> Dict[str, Any]:
    target_id = get_target_id(subgraph)
    adjacency, all_nodes = _build_adjacency_from_subgraph(subgraph)
    n = len(all_nodes)
    if n <= 1:
        return {
            "target_node": {
                "id": target_id,
                "degree": 0,
                "degree_centrality": 0.0,
                "clustering_coefficient": 0.0,
                "betweenness_centrality": 0.0,
                "core_number": 0
            },
            "global": {
                "total_nodes": n,
                "total_edges": 0,
                "avg_degree": 0.0,
                "avg_clustering_coefficient": 0.0,
                "max_core_number": 0,
                "density": 0.0
            },
            "top_nodes_by_degree": [],
            "top_nodes_by_betweenness": [],
            "degree_distribution": {}
        }

    degrees = _compute_degrees(adjacency, all_nodes)
    degree_centrality = {node: degrees[node] / (n - 1) for node in all_nodes}

    clustering = {}
    for node in all_nodes:
        clustering[node] = _compute_clustering_coefficient(adjacency, node)

    betweenness = _compute_betweenness_centrality(adjacency, all_nodes)

    core_numbers = _compute_kcore(adjacency, all_nodes)
    total_edges = sum(degrees.values()) // 2
    avg_degree = sum(degrees.values()) / n if n > 0 else 0.0
    avg_clustering = sum(clustering.values()) / n if n > 0 else 0.0
    max_core = max(core_numbers.values()) if core_numbers else 0
    max_possible_edges = n * (n - 1) / 2
    density = total_edges / max_possible_edges if max_possible_edges > 0 else 0.0

    sorted_by_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    top_by_degree = [{"id": nid, "degree": deg} for nid, deg in sorted_by_degree]

    sorted_by_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
    top_by_betweenness = [{"id": nid, "betweenness": round(bc, 4)} for nid, bc in sorted_by_betweenness]
    degree_values = list(degrees.values())
    degree_distribution = {
        "min": min(degree_values) if degree_values else 0,
        "max": max(degree_values) if degree_values else 0,
        "mean": round(avg_degree, 2),
        "median": sorted(degree_values)[len(degree_values) // 2] if degree_values else 0
    }
    target_degree = degrees.get(target_id, 0)
    target_dc = degree_centrality.get(target_id, 0.0)
    target_cc = clustering.get(target_id, 0.0)
    target_bc = betweenness.get(target_id, 0.0)
    target_core = core_numbers.get(target_id, 0)
    return {
        "target_node": {
            "id": target_id,
            "degree": target_degree,
            "degree_centrality": round(target_dc, 4),
            "clustering_coefficient": round(target_cc, 4),
            "betweenness_centrality": round(target_bc, 4),
            "core_number": target_core
        },
        "global": {
            "total_nodes": n,
            "total_edges": total_edges,
            "avg_degree": round(avg_degree, 2),
            "avg_clustering_coefficient": round(avg_clustering, 4),
            "max_core_number": max_core,
            "density": round(density, 4)
        },
        "top_nodes_by_degree": top_by_degree,
        "top_nodes_by_betweenness": top_by_betweenness,
        "degree_distribution": degree_distribution
    }


def build_round1_prompt(subgraph: Dict, graph_stats: Dict) -> str:
    return build_shared_round1_prompt(subgraph, graph_stats)


def build_round2_prompt(subgraph: Dict, graph_stats: Dict, round1_response: str) -> str:
    return build_shared_round2_prompt(subgraph, graph_stats, round1_response)


def build_final_system_prompt() -> str:
    return CONTEXTUAL_FEATURE_SYSTEM_PROMPT


def build_final_user_prompt(subgraph: Dict, graph_stats: Dict, multi_round_context: List[str]) -> str:
    return build_shared_final_user_prompt(subgraph, graph_stats, multi_round_context)


def call_api_multiround(
    messages: List[Dict],
    model: str = MODEL_NAME,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float = 0.9
) -> Tuple[str, int]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=API_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        tokens_used = result.get("usage", {}).get("total_tokens", 0)
        if VERBOSE_LLM:
            print(f"    [LLM RESPONSE] (tokens={tokens_used}) {content[:2000]}")

        return content, tokens_used
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise
    except KeyError as e:
        logger.error(f"API response parsing failed: {e}")
        logger.error(f"API response: {response.text}")
        raise


class MultiRoundRiskAnalyzer:
    def __init__(self, subgraph: Dict, target_account: str, graph_stats: Dict = None,
                 model: str = MODEL_NAME, temperature: float = 0.9):
        self.subgraph = subgraph
        self.target_account = target_account
        self.graph_stats = graph_stats or {}
        self.model = model
        self.temperature = temperature
        self.multi_round_context = []

    def run_round1(self) -> str:
        logger.info(f"[{self.target_account}] Starting Round 1: statistic interpretation and risk signals")
        user_prompt = build_round1_prompt(self.subgraph, self.graph_stats)
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        response, tokens_used = call_api_multiround(
            messages, model=self.model, temperature=self.temperature)
        logger.info(f"[{self.target_account}] Round 1 completed, tokens used: {tokens_used}")
        logger.debug(f"[{self.target_account}] Round 1 response:\n{response}")
        self.multi_round_context.append(response)
        return response

    def run_round2(self, round1_response: str) -> str:
        logger.info(f"[{self.target_account}] Starting Round 2: targeted deep dive")
        user_prompt = build_round2_prompt(self.subgraph, self.graph_stats, round1_response)
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        response, tokens_used = call_api_multiround(
            messages, model=self.model, temperature=self.temperature)
        logger.info(f"[{self.target_account}] Round 2 completed, tokens used: {tokens_used}")
        logger.debug(f"[{self.target_account}] Round 2 response:\n{response}")
        self.multi_round_context.append(response)
        return response

    def run_final_round(self) -> str:
        logger.info(f"[{self.target_account}] Starting final round: comprehensive output")
        system_prompt = build_final_system_prompt()
        user_prompt = build_final_user_prompt(self.subgraph, self.graph_stats, self.multi_round_context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response, tokens_used = call_api_multiround(
            messages, model=self.model, temperature=self.temperature, max_tokens=16000)
        logger.info(f"[{self.target_account}] Final Round completed, tokens used: {tokens_used}")
        logger.debug(f"[{self.target_account}] Final Round response:\n{response}")
        return response

    def run_full_analysis(self) -> Dict:
        logger.info(f"[{self.target_account}] Starting multi-round risk analysis")
        round1_response = self.run_round1()

        round2_response = self.run_round2(round1_response)
        final_response = self.run_final_round()
        result = {
            "target_account": self.target_account,
            "analysis_method": "multi_round",
            "rounds": [
                {
                    "round": 1,
                    "focus": "statistic interpretation and risk-signal identification",
                    "response": _prettify_json_response(round1_response).split("\n") if isinstance(round1_response, str) else round1_response
                },
                {
                    "round": 2,
                    "focus": "targeted deep dive",
                    "response": _prettify_json_response(round2_response).split("\n") if isinstance(round2_response, str) else round2_response
                }
            ],
            "final_analysis": _prettify_json_response(final_response).split("\n") if isinstance(final_response, str) else final_response,
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"[{self.target_account}] Multi-round risk analysis completed")
        return result


def process_single_subgraph(subgraph: Dict[str, Any], index: int,
                           model: str = MODEL_NAME,
                           temperature: float = 0.9,
                           enable_pruning: bool = None) -> Dict[str, Any]:
    target_id = get_target_id(subgraph)
    entity_name = get_target_entity_name(subgraph)
    _original = subgraph.get("english_subgraph")
    print(f"  [{index}] Target: {target_id} ({entity_name})")
    if is_trivial_subgraph(subgraph):
        print("    [SKIP] No neighborhood expansion")
        return {
            "index": index,
            "target_id": target_id,
            "entity_name": entity_name,
            "status": "skipped",
            "reason": "subgraph contains only the target node"
        }
    if enable_pruning is None:
        enable_pruning = ENABLE_PRUNING
    english_subgraph_from_format = None
    if isinstance(subgraph, dict) and 'english_subgraph' in subgraph:
        english_subgraph_from_format = subgraph.pop('english_subgraph')
        print(f"    Detached raw subgraph from formatter: "
              f"{len(english_subgraph_from_format.get('nodes', []))} nodes, "
              f"{len(english_subgraph_from_format.get('edges', []))} edges")
    formatted_subgraph = subgraph
    if enable_pruning:
        print("    Pruning subgraph...")
        original_tokens = estimate_subgraph_tokens(formatted_subgraph)
        formatted_subgraph = prune_subgraph(formatted_subgraph, RELATION_IMPORTANCE)
        pruned_tokens = estimate_subgraph_tokens(formatted_subgraph)
        saved_tokens = original_tokens - pruned_tokens
        print(f"    Original tokens: {original_tokens}, after pruning: {pruned_tokens}, saved: {saved_tokens}")
    english_subgraph = None
    source_for_english = _original if (_original and isinstance(_original, dict)) else english_subgraph_from_format
    if source_for_english and isinstance(source_for_english, dict):
        kept_node_ids = _collect_node_ids(formatted_subgraph)
        english_subgraph = _filter_original_by_kept_ids(source_for_english, kept_node_ids)
        print(f"    Raw subgraph after aligned filtering: {len(english_subgraph.get('nodes', []))} nodes, "
              f"{len(english_subgraph.get('edges', []))} edges")
    print("    Computing graph statistics...")
    graph_stats = compute_graph_statistics(formatted_subgraph)
    target_stats = graph_stats.get("target_node", {})
    print(f"    Target node: degree={target_stats.get('degree', 0)}, "
          f"clustering={target_stats.get('clustering_coefficient', 0):.3f}, "
          f"betweenness={target_stats.get('betweenness_centrality', 0):.3f}, "
          f"core={target_stats.get('core_number', 0)}")

    print("    Running multi-round analysis...")
    t0 = time.time()
    analyzer = MultiRoundRiskAnalyzer(
        formatted_subgraph, target_id, graph_stats=graph_stats,
        model=model, temperature=temperature)
    analysis_result = analyzer.run_full_analysis()

    elapsed = time.time() - t0
    print(f"    Completed ({elapsed:.1f}s)")
    result = {
        "index": index,
        "target_id": target_id,
        "entity_name": entity_name,
        "status": "success",
        "model": model,
        "temperature": temperature,
        "elapsed_seconds": round(elapsed, 1),
        "analysis_result": analysis_result,
        "formatted_subgraph": formatted_subgraph,
        "graph_stats": graph_stats,
    }
    if english_subgraph:
        result["english_subgraph"] = english_subgraph
    return result
