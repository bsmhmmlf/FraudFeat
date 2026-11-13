import copy

import json

import math

import os

import re

import time

from collections import defaultdict

from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from prompt_common import CONTEXTUAL_FEATURE_SYSTEM_PROMPT, build_comm_l1_prompt as build_shared_l1_prompt, build_comm_l3_user_prompt as build_shared_l3_user_prompt


API_URL = os.environ.get("SYSTEM_LLM_API_URL", "<YOUR_LLM_API_URL>")
DEFAULT_API_KEY = os.environ.get("SYSTEM_LLM_API_KEY", "<YOUR_LLM_API_KEY>")
HEADERS = {
    "Authorization": "Bearer {}".format(DEFAULT_API_KEY),
    "Content-Type": "application/json",
}
DEFAULT_MODEL = "hy4-preview"

API_TIMEOUT = 600

VERBOSE_LLM = True

MAX_SUPER_COMMUNITIES = 4

LLM_SUMMARY_TOP_N = 12

MAX_COMMUNITY_RATIO = 0.5

MERGE_SINGLETON_COMMUNITIES = True

ENABLE_PRUNING = True

FIELD_THRESHOLD = 60

FIELD_IMPORTANCE: Dict[str, float] = {}

RELATION_IMPORTANCE: Dict = {}
TARGET_SUBGRAPH_TOKENS = 96000
BLACKLIST_TYPES = set()

SYSTEM_PROMPT = CONTEXTUAL_FEATURE_SYSTEM_PROMPT


def call_api(system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL,
             max_retries: int = 3, temperature: float = 0.6) -> Optional[str]:
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_completion_tokens": 81920
    }
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(API_URL, headers=HEADERS,
                                     data=json.dumps(data), timeout=API_TIMEOUT)
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                if VERBOSE_LLM:
                    print(f"    [LLM RESPONSE] {content[:2000]}")
                return content
            elif response.status_code == 429:
                wait = 5 * attempt
                print(f"    [RATE LIMIT] Retrying in {wait}s...")
                time.sleep(wait)
                continue
            else:
                print(f"    [ERROR] API returned {response.status_code}")
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                return None
        except requests.exceptions.Timeout:
            print(f"    [TIMEOUT] Request timed out (attempt {attempt}/{max_retries})")
            if attempt < max_retries:
                time.sleep(3)
                continue
            return None
        except Exception as e:
            print(f"    [ERROR] API request failed: {e}")
            if attempt < max_retries:
                time.sleep(2)
                continue
            return None
    return None


def estimate_tokens_simple(text: str) -> int:
    chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    others = len(text) - chinese
    return int(chinese * 2 + others * 0.25)


def estimate_subgraph_tokens(subgraph: Dict[str, Any]) -> int:
    subgraph_json = json.dumps(subgraph, ensure_ascii=False, indent=2)
    return estimate_tokens_simple(subgraph_json)


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
    mutable_nodes = []
    for node_id, score in importance.items():
        if score != float('inf'):
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
    return {"nodes": filtered_nodes, "edges": filtered_edges}


def is_trivial_subgraph(subgraph: Dict[str, Any]) -> bool:
    has_groups = len(subgraph.get("relation_groups", [])) > 0
    has_members = len(subgraph.get("member_attributes", [])) > 0
    has_diff_rels = len(subgraph.get("relations", [])) > 0
    return not (has_groups or has_members or has_diff_rels)


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


def _build_node_type_map(subgraph: Dict[str, Any]) -> Dict[str, str]:
    subgraph = _en_anon_view(subgraph)
    node_type: Dict[str, str] = {}
    for m in subgraph.get("member_attributes", []):
        nid = m.get("id", "")
        en = m.get("entity", "")
        if nid:
            node_type[nid] = en
    target = subgraph.get("target_account", {})
    if isinstance(target, dict):
        nid = target.get("id", "")
        if nid:
            node_type[nid] = target.get("entity", "")
    return node_type


def _identify_black_nodes(subgraph: Dict[str, Any]) -> Set[str]:
    node_type = _build_node_type_map(subgraph)
    return {nid for nid, en in node_type.items() if en in BLACKLIST_TYPES}


def _local_move(
    adj: Dict[str, Dict[str, float]],
    deg: Dict[str, float],
    m2: float,
    nodes: List[str],
    comm: Dict[str, str],
    resolution: float = 1.0,
) -> bool:
    comm_tot: Dict[str, float] = defaultdict(float)
    for n in nodes:
        comm_tot[comm[n]] += deg[n]
    moved_any = False
    changed = True
    iters = 0
    while changed and iters < 50:
        changed = False
        iters += 1
        for n in nodes:
            cn = comm[n]
            k_i = deg[n]
            neigh_comm: Dict[str, float] = defaultdict(float)
            for nb, w in adj.get(n, {}).items():
                if nb in comm:
                    neigh_comm[comm[nb]] += w
            comm_tot[cn] -= k_i
            best_c = cn
            best_gain = neigh_comm.get(cn, 0.0) - resolution * comm_tot[cn] * k_i / m2
            for c, k_i_in in neigh_comm.items():
                if c == cn:
                    continue
                gain = k_i_in - resolution * comm_tot[c] * k_i / m2
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_c = c
            comm_tot[best_c] += k_i
            if best_c != cn:
                comm[n] = best_c
                changed = True
                moved_any = True
    return moved_any


def _connected_components(
    adj: Dict[str, Dict[str, float]],
    members: List[str],
    member_set: Set[str],
) -> List[List[str]]:
    visited: Set[str] = set()
    comps: List[List[str]] = []
    for start in members:
        if start in visited:
            continue
        comp: List[str] = []
        stack = [start]
        visited.add(start)
        while stack:
            v = stack.pop()
            comp.append(v)
            for nb in adj.get(v, {}):
                if nb in member_set and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        comps.append(comp)
    return comps


def _refine_partition(
    adj: Dict[str, Dict[str, float]],
    deg: Dict[str, float],
    m2: float,
    nodes: List[str],
    comm: Dict[str, str],
    resolution: float = 1.0,
) -> Dict[str, str]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for n in nodes:
        groups[comm[n]].append(n)
    refined: Dict[str, str] = {}
    rid = 0
    for members in groups.values():
        member_set = set(members)
        for comp in _connected_components(adj, members, member_set):
            if len(comp) == 1:
                refined[comp[0]] = f"r{rid}"
                rid += 1
                continue
            comp_set = set(comp)

            sub_comm: Dict[str, str] = {n: f"r{rid + i}" for i, n in enumerate(comp)}
            rid += len(comp)
            sub_tot: Dict[str, float] = defaultdict(float)
            for n in comp:
                sub_tot[sub_comm[n]] += deg[n]
            changed = True
            it = 0
            while changed and it < 50:
                changed = False
                it += 1
                for n in comp:
                    cn = sub_comm[n]
                    k_i = deg[n]
                    neigh_comm: Dict[str, float] = defaultdict(float)
                    for nb, w in adj.get(n, {}).items():
                        if nb in comp_set:
                            neigh_comm[sub_comm[nb]] += w
                    sub_tot[cn] -= k_i
                    best_c = cn
                    best_gain = neigh_comm.get(cn, 0.0) - resolution * sub_tot[cn] * k_i / m2
                    for cc, k_i_in in neigh_comm.items():
                        if cc == cn:
                            continue
                        gain = k_i_in - resolution * sub_tot[cc] * k_i / m2
                        if gain > best_gain + 1e-12:
                            best_gain = gain
                            best_c = cc
                    sub_tot[best_c] += k_i
                    if best_c != cn:
                        sub_comm[n] = best_c
                        changed = True
            sub_groups: Dict[str, List[str]] = defaultdict(list)
            for n in comp:
                sub_groups[sub_comm[n]].append(n)
            for sg_members in sub_groups.values():
                sg_set = set(sg_members)
                for cc2 in _connected_components(adj, sg_members, sg_set):
                    for n in cc2:
                        refined[n] = f"r{rid}"
                    rid += 1
    return refined


def _project_to_original(
    comm: Dict[str, str],
    cur_nodes: List[str],
    super_members: Dict[str, List[str]],
) -> List[List[str]]:
    grp: Dict[str, List[str]] = defaultdict(list)
    for sn in cur_nodes:
        grp[comm[sn]].extend(super_members.get(sn, [sn]))
    comms = [v for v in grp.values() if v]
    comms.sort(key=len, reverse=True)
    return comms


def leiden_communities(
    adjacency: Dict[str, Dict[str, float]],
    nodes: List[str],
    max_levels: int = 5,
    resolution: float = 1.0
) -> List[List[str]]:
    if not nodes:
        return []
    adj0: Dict[str, Dict[str, float]] = {}
    for k, v in adjacency.items():
        sk = str(k)
        adj0[sk] = {}
        for k2, v2 in v.items():
            adj0[sk][str(k2)] = float(v2)
    work_nodes = [str(n) for n in nodes]
    cur_adj = copy.deepcopy(adj0)
    cur_nodes = list(work_nodes)
    super_members: Dict[str, List[str]] = {n: [n] for n in work_nodes}
    init_comm: Dict[str, str] = {n: n for n in work_nodes}
    best: List[List[str]] = [[n] for n in work_nodes]
    for _level in range(max_levels):
        deg: Dict[str, float] = {}
        m2 = 0.0
        for n in cur_nodes:
            d = sum(cur_adj.get(n, {}).values())
            deg[n] = d
            m2 += d
        if m2 == 0:
            break

        comm = dict(init_comm)
        moved = _local_move(cur_adj, deg, m2, cur_nodes, comm, resolution)

        best = _project_to_original(comm, cur_nodes, super_members)
        if not moved:
            break
        if len(set(comm.values())) <= 1:
            break
        refined = _refine_partition(cur_adj, deg, m2, cur_nodes, comm, resolution)
        refined_ids = sorted(set(refined.values()))

        if len(refined_ids) >= len(cur_nodes):
            break
        new_adj: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for src in cur_nodes:
            rs = refined[src]
            for dst, w in cur_adj.get(src, {}).items():
                rd = refined[dst]
                if rs != rd:
                    new_adj[rs][rd] += w
        new_adj_clean: Dict[str, Dict[str, float]] = {r: {} for r in refined_ids}
        for r, nbrs in new_adj.items():
            new_adj_clean[r] = dict(nbrs)
        new_super_members: Dict[str, List[str]] = defaultdict(list)
        for n in cur_nodes:
            new_super_members[refined[n]].extend(super_members.get(n, [n]))
        new_init_comm: Dict[str, str] = {}
        for n in cur_nodes:
            new_init_comm.setdefault(refined[n], comm[n])
        cur_adj = new_adj_clean
        cur_nodes = refined_ids
        super_members = dict(new_super_members)
        init_comm = new_init_comm
    return best if best else [list(work_nodes)]


def _partition_modularity(
    adj: Dict[str, Dict[str, float]],
    communities: List[List[str]],
    resolution: float = 1.0
) -> float:
    deg: Dict[str, float] = {}
    m2 = 0.0
    for n, nbrs in adj.items():
        d = sum(nbrs.values())
        deg[n] = d
        m2 += d
    if m2 == 0:
        return 0.0
    node2c: Dict[str, int] = {}
    for i, c in enumerate(communities):
        for n in c:
            node2c[n] = i
    in_w: Dict[int, float] = defaultdict(float)
    tot: Dict[int, float] = defaultdict(float)
    for n, nbrs in adj.items():
        cn = node2c.get(n)
        if cn is None:
            continue
        tot[cn] += deg.get(n, 0.0)
        for nb, w in nbrs.items():
            if node2c.get(nb) == cn:
                in_w[cn] += w
    q = 0.0
    for c in set(node2c.values()):
        q += in_w[c] / m2 - resolution * (tot[c] / m2) ** 2
    return q


def _split_large_communities(
    adjacency: Dict[str, Dict[str, float]],
    communities: List[List[str]],
    max_ratio: float = MAX_COMMUNITY_RATIO,
    resolution: float = 2.0,
    min_gain: float = 1e-4
) -> List[List[str]]:
    total = sum(len(c) for c in communities)
    if total == 0:
        return communities
    result: List[List[str]] = []
    for comm in communities:
        if len(comm) <= 2 or len(comm) / total <= max_ratio:
            result.append(comm)
            continue
        comm_set = set(comm)
        sub_adj: Dict[str, Dict[str, float]] = {}
        for n in comm:
            sub_adj[n] = {
                nb: w for nb, w in adjacency.get(n, {}).items()
                if nb in comm_set
            }
        sub_comms = leiden_communities(sub_adj, list(comm), max_levels=3,
                                       resolution=resolution)
        sub_comms = [c for c in sub_comms if c]
        if len(sub_comms) <= 1:
            result.append(comm)
            continue
        q_split = _partition_modularity(sub_adj, sub_comms, resolution=1.0)
        q_whole = _partition_modularity(sub_adj, [list(comm)], resolution=1.0)
        if q_split - q_whole > min_gain:
            result.extend(sub_comms)
        else:
            print(f"      Large community (size={len(comm)}) has insufficient splitting gain "
                  f"(delta_Q={q_split - q_whole:.4f}); retaining one community")
            result.append(comm)
    result.sort(key=len, reverse=True)
    return result


def _merge_singleton_communities(
    adjacency: Dict[str, Dict[str, float]],
    communities: List[List[str]]
) -> List[List[str]]:
    if len(communities) <= 1:
        return communities
    big = [c for c in communities if len(c) > 1]
    singletons = [c for c in communities if len(c) == 1]
    if not singletons or not big:
        return communities

    node2comm: Dict[str, int] = {}
    for i, c in enumerate(big):
        for n in c:
            node2comm[n] = i
    merged = [list(c) for c in big]
    unmerged_singletons: List[List[str]] = []
    for s in singletons:
        node = s[0]
        comm_weights: Dict[int, float] = defaultdict(float)
        for nb, w in adjacency.get(node, {}).items():
            ci = node2comm.get(nb)
            if ci is not None and nb != node:
                comm_weights[ci] += w
        if comm_weights:
            best_comm = max(comm_weights, key=comm_weights.get)
            merged[best_comm].append(node)
        else:
            unmerged_singletons.append(s)
    merged.extend(unmerged_singletons)
    merged.sort(key=len, reverse=True)
    return merged


def _build_community_adjacency(
    adjacency: Dict[str, Dict[str, float]],
    communities: List[List[str]]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    node2comm: Dict[str, str] = {}
    for i, comm in enumerate(communities):
        ci = str(i)
        for n in comm:
            node2comm[str(n)] = ci
    comm_adj: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for src, nbrs in adjacency.items():
        ssrc = str(src)
        if ssrc not in node2comm:
            continue
        c_src = node2comm[ssrc]
        for dst, w in nbrs.items():
            sdst = str(dst)
            if sdst not in node2comm:
                continue
            c_dst = node2comm[sdst]
            if c_src != c_dst:
                comm_adj[c_src][c_dst] += w
                comm_adj[c_dst][c_src] += w
    return {k: dict(v) for k, v in comm_adj.items()}, node2comm


def merge_communities(
    adjacency: Dict[str, Dict[str, float]],
    communities: List[List[str]],
    max_super: int = MAX_SUPER_COMMUNITIES
) -> List[List[str]]:
    if len(communities) <= max_super:
        return [list(c) for c in communities]
    comm_adj, node2comm = _build_community_adjacency(adjacency, communities)
    comm_nodes = [str(i) for i in range(len(communities))]
    super_comms = leiden_communities(comm_adj, comm_nodes, max_levels=3)

    while len(super_comms) > max_super:
        super_comms.sort(key=len)
        a = super_comms.pop(0)
        b = super_comms.pop(0)
        super_comms.append(a + b)
    result: List[List[str]] = []
    for sc in super_comms:
        orig_nodes: List[str] = []
        for comm_id_str in sc:
            ci = int(comm_id_str)
            orig_nodes.extend(communities[ci])
        if orig_nodes:
            result.append(orig_nodes)
    result.sort(key=len, reverse=True)
    return result


def slice_community_subgraph(
    subgraph: Dict[str, Any],
    target_id: str,
    community_nodes: List[str]
) -> Dict[str, Any]:
    keep = set(community_nodes)
    keep.add(target_id)
    return _keep_by_ids(subgraph, keep, keep=True)


def _shortest_hop_to_target(
    adjacency: Dict[str, Dict[str, float]],
    target: str,
    dests: Set[str]
) -> Dict[str, int]:
    if not dests:
        return {}
    dist = {target: 0}
    queue = [target]
    qi = 0
    while qi < len(queue):
        v = queue[qi]
        qi += 1
        for nb in adjacency.get(v, {}):
            if nb not in dist:
                dist[nb] = dist[v] + 1
                queue.append(nb)
                if len(dist) > len(dests) * 10 + 100:
                    break
    return {d: dist.get(d, -1) for d in dests if d in dist}


def profile_communities(
    subgraph: Dict[str, Any],
    communities: List[List[str]],
    betweenness: Dict[str, float],
    black_nodes: Set[str],
    target_id: str
) -> List[Dict[str, Any]]:
    adjacency, _ = _build_adjacency_from_subgraph(subgraph)
    profiles = []
    for i, comm in enumerate(communities):
        comm_set = set(comm)
        blacks_in = comm_set & black_nodes

        hop_map = _shortest_hop_to_target(adjacency, target_id, blacks_in)
        min_hop = min(hop_map.values()) if hop_map else 999
        max_bc = max(
            (betweenness.get(n, 0.0) for n in comm if n in betweenness),
            default=0.0
        )
        profiles.append({
            "id": i,
            "size": len(comm),
            "black_count": len(blacks_in),
            "black_nodes": sorted(blacks_in),
            "min_hop_to_target": min_hop,
            "max_betweenness": round(max_bc, 4),
            "members": comm,
        })
    return profiles


def select_top_communities(
    profiles: List[Dict[str, Any]],
    top_n: int = LLM_SUMMARY_TOP_N
) -> List[int]:
    scored = []
    for p in profiles:
        score = 0
        if p["black_count"] > 0:
            score += 1000
        score += (10 - min(p["min_hop_to_target"], 10)) * 50
        score += p["max_betweenness"] * 100
        scored.append((p["id"], score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in scored[:top_n]]


def structured_community_summary(
    subgraph: Dict[str, Any],
    community: List[str],
    profile: Dict[str, Any],
    target_id: str
) -> Dict[str, Any]:
    node_type = _build_node_type_map(subgraph)
    type_dist: Dict[str, int] = defaultdict(int)
    for n in community:
        type_dist[node_type.get(n, "unknown")] += 1
    return {
        "community_id": profile["id"],
        "summary_method": "code_structured",
        "size": profile["size"],
        "entity_type_distribution": dict(type_dist),
        "black_label_count": profile["black_count"],
        "black_nodes": profile["black_nodes"],
        "min_hop_to_target": profile["min_hop_to_target"],
        "max_betweenness": profile["max_betweenness"],
        "note": "Deterministic structured summary (no LLM call)"
    }


def build_l1_prompt(comm_subgraph: Dict[str, Any], target_id: str, profile: Dict[str, Any]) -> str:
    return build_shared_l1_prompt(comm_subgraph, target_id, profile)


def build_l2_prompt(super_id: int, l1_summaries: List[Dict[str, Any]], comm_connections: str) -> str:
    from prompt_common import build_comm_l2_prompt
    return build_comm_l2_prompt(super_id, l1_summaries, comm_connections)


def build_l3_user_prompt(subgraph: Dict[str, Any], target_id: str, l2_summaries: List[Dict[str, Any]], l1_summaries: List[Dict[str, Any]]) -> str:
    return build_shared_l3_user_prompt(subgraph, target_id, l2_summaries, l1_summaries)


class GraphRAGAnalyzer:
    def __init__(self, subgraph: Dict[str, Any], target_id: str,
                 model: str = DEFAULT_MODEL, temperature: float = 0.9):
        self.subgraph = subgraph
        self.target_id = target_id
        self.model = model
        self.temperature = temperature
        self.l1_summaries: List[Dict[str, Any]] = []
        self.l2_summaries: List[Dict[str, Any]] = []
        self.communities: List[List[str]] = []
        self.super_communities: List[List[str]] = []
        self._target_neighbors: Set[str] = set()
        self._adjacency: Dict[str, Dict[str, float]] = {}

    def _detect_communities(self) -> Tuple[List[List[str]], List[List[str]], Set[str]]:
        adjacency, all_nodes = _build_adjacency_from_subgraph(self.subgraph)
        black_nodes = _identify_black_nodes(self.subgraph)
        neighbor_nodes = [n for n in all_nodes if n != self.target_id]
        if len(neighbor_nodes) <= 1:
            return [neighbor_nodes], [neighbor_nodes], black_nodes
        neighbor_set = set(neighbor_nodes)
        filtered_adj: Dict[str, Dict[str, float]] = {}
        for n in neighbor_nodes:
            filtered_adj[n] = {
                nb: w for nb, w in adjacency.get(n, {}).items()
                if nb in neighbor_set
            }
        n_nodes = len(neighbor_nodes)
        n_edges = sum(len(nbrs) for nbrs in filtered_adj.values()) / 2.0
        max_edges = n_nodes * (n_nodes - 1) / 2.0
        density = n_edges / max_edges if max_edges > 0 else 0.0
        resolution = max(1.5, min(1.0 + density * 2.5, 3.0))
        print(f"    Graph density={density:.3f} ({int(n_edges)} edges/{n_nodes} nodes), "
              f"adaptive resolution={resolution:.2f}")
        l1 = leiden_communities(filtered_adj, neighbor_nodes,
                                max_levels=3, resolution=resolution)
        l1 = [c for c in l1 if c]
        l1 = _split_large_communities(filtered_adj, l1,
                                       MAX_COMMUNITY_RATIO, resolution * 1.5)
        if MERGE_SINGLETON_COMMUNITIES:
            l1 = _merge_singleton_communities(filtered_adj, l1)
        l2 = merge_communities(adjacency, l1, MAX_SUPER_COMMUNITIES)
        return l1, l2, black_nodes

    def _find_bridge_nodes(self, community_nodes: List[str]) -> List[str]:
        target = self.target_id
        comm_set = set(community_nodes)
        adj = self._adjacency
        dist = {target: 0}
        pred = {target: None}
        queue = [target]
        qi = 0
        while qi < len(queue):
            v = queue[qi]
            qi += 1
            for nb in adj.get(v, {}):
                if nb not in dist:
                    dist[nb] = dist[v] + 1
                    pred[nb] = v
                    queue.append(nb)
        bridge_nodes = set()
        for member in community_nodes:
            if member in dist and member != target:
                cur = pred.get(member)
                while cur is not None and cur != target:
                    if cur not in comm_set:
                        bridge_nodes.add(cur)
                    cur = pred.get(cur)
        return list(bridge_nodes)

    def _summarize_l1(self, profiles: List[Dict[str, Any]],
                      top_n_ids: List[int]) -> List[Dict[str, Any]]:
        summaries = []
        for p in profiles:
            if p["id"] in top_n_ids:
                comm_members = set(p["members"])
                has_direct_conn = bool(comm_members & self._target_neighbors)
                members_for_slice = p["members"]
                if not has_direct_conn:
                    bridge_nodes = self._find_bridge_nodes(p["members"])
                    if bridge_nodes:
                        members_for_slice = p["members"] + bridge_nodes
                        print(f"      L1 community {p['id']} has no direct target connection; "
                              f"adding {len(bridge_nodes)} bridge nodes")
                    else:
                        print(f"      [WARN] L1 community {p['id']} is disconnected from the target; "
                              f"the target is isolated in the slice")
                comm_sg = slice_community_subgraph(
                    self.subgraph, self.target_id, members_for_slice
                )
                prompt = build_l1_prompt(comm_sg, self.target_id, p)
                print(f"      Calling LLM for L1 community {p['id']} (size={p['size']}, "
                      f"black={p['black_count']})...")
                resp = call_api("", prompt, model=self.model,
                                temperature=self.temperature)
                try:
                    card = json.loads(resp.strip().strip('`').strip())
                    card["summary_method"] = "llm"
                    summaries.append(card)
                except Exception:
                    summaries.append(structured_community_summary(
                        self.subgraph, p["members"], p, self.target_id
                    ))
                time.sleep(0.5)
            else:
                summaries.append(structured_community_summary(
                    self.subgraph, p["members"], p, self.target_id
                ))
        return summaries

    def _summarize_l2(self, l1_summaries: List[Dict[str, Any]],
                      l1_profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        node2l1: Dict[str, int] = {}
        for p in l1_profiles:
            for n in p["members"]:
                node2l1[n] = p["id"]
        summaries = []
        for sid, super_comm in enumerate(self.super_communities):
            l1_ids_in = set()
            for n in super_comm:
                if n in node2l1:
                    l1_ids_in.add(node2l1[n])
            cards = [s for s in l1_summaries
                     if s.get("community_id") in l1_ids_in]
            conn_lines = []
            for i in l1_ids_in:
                for j in l1_ids_in:
                    if i < j:
                        conn_lines.append(f"Community {i} <-> Community {j}")
            conn_str = "\n".join(conn_lines) if conn_lines else "(no direct inter-community connections)"
            print(f"      Calling LLM for L2 super-community {sid} "
                  f"({len(cards)} subcommunities)...")
            prompt = build_l2_prompt(sid, cards, conn_str)
            resp = call_api("", prompt, model=self.model,
                            temperature=self.temperature)
            try:
                card = json.loads(resp.strip().strip('`').strip())
                summaries.append(card)
            except Exception:
                summaries.append({
                    "super_community_id": sid,
                    "summary": f"LLM summary parsing failed for super-community {sid} ({len(cards)} subcommunities)",
                    "most_suspicious_subcommunities": [],
                    "pattern_tendency": "unknown"
                })
            time.sleep(0.5)
        return summaries

    def run_full_analysis(self) -> str:
        adjacency, all_nodes = _build_adjacency_from_subgraph(self.subgraph)
        betweenness = _compute_betweenness_centrality(adjacency, all_nodes)
        self._target_neighbors = set(adjacency.get(self.target_id, {}).keys())
        self._adjacency = adjacency
        print("    Detecting communities...")
        l1, l2, black_nodes = self._detect_communities()
        self.communities = l1
        self.super_communities = l2
        print(f"    L1 communities: {len(l1)}, L2 super-communities: {len(l2)}, "
              f"black-label nodes: {len(black_nodes)}")
        profiles = profile_communities(
            self.subgraph, l1, betweenness, black_nodes, self.target_id
        )
        top_n_ids = select_top_communities(profiles, LLM_SUMMARY_TOP_N)
        print(f"    Top-N communities selected for LLM summaries: {top_n_ids}")
        print("    Summarizing L1 communities...")
        self.l1_summaries = self._summarize_l1(profiles, top_n_ids)

        print("    Summarizing L2 super-communities...")
        self.l2_summaries = self._summarize_l2(self.l1_summaries, profiles)
        print("    L3 global synthesis...")
        user_prompt = build_l3_user_prompt(
            self.subgraph, self.target_id,
            self.l2_summaries, self.l1_summaries
        )
        final = call_api(SYSTEM_PROMPT, user_prompt, model=self.model,
                         temperature=self.temperature, max_retries=3)
        return final


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


def process_subgraph_graphrag(
    subgraph: Dict[str, Any], index: int,
    model: str = DEFAULT_MODEL, temperature: float = 0.9,
    enable_pruning: bool = None
) -> Dict[str, Any]:
    target_id = get_target_id(subgraph)
    entity_name = get_target_entity_name(subgraph)
    _original = subgraph.get("english_subgraph")
    print(f"  [{index}] Target: {target_id} ({entity_name})")
    if is_trivial_subgraph(subgraph):
        print("    [SKIP] No neighborhood expansion")
        return {
            "index": index, "target_id": target_id,
            "entity_name": entity_name, "status": "skipped",
            "reason": "subgraph contains only the target node"
        }

    if enable_pruning is None:
        enable_pruning = ENABLE_PRUNING
    english_from_format = None
    if isinstance(subgraph, dict) and "english_subgraph" in subgraph:
        english_from_format = subgraph.pop("english_subgraph")
        print(f"    Detached raw subgraph: "
              f"{len(english_from_format.get('nodes', []))} nodes, "
              f"{len(english_from_format.get('edges', []))} edges")
    formatted_subgraph = subgraph
    if enable_pruning:
        print("    Pruning subgraph...")
        orig_tok = estimate_subgraph_tokens(formatted_subgraph)
        formatted_subgraph = prune_subgraph(formatted_subgraph, RELATION_IMPORTANCE)
        pruned_tok = estimate_subgraph_tokens(formatted_subgraph)
        print(f"    Original tokens: {orig_tok}, after pruning: {pruned_tok}")
    english_subgraph = None
    src = _original if (_original and isinstance(_original, dict)) else english_from_format
    if src and isinstance(src, dict):
        kept = _collect_node_ids(formatted_subgraph)
        english_subgraph = _filter_original_by_kept_ids(src, kept)
    print("    GraphRAG hierarchical community analysis...")
    t0 = time.time()
    analyzer = GraphRAGAnalyzer(formatted_subgraph, target_id, model=model,
                                temperature=temperature)
    try:
        analysis = analyzer.run_full_analysis()
    except Exception as e:
        print(f"    [ERROR] GraphRAG analysis failed: {e}")
        return {
            "index": index, "target_id": target_id,
            "entity_name": entity_name, "status": "error",
            "reason": f"GraphRAG analysis failed: {e}"
        }
    elapsed = time.time() - t0
    print(f"    Completed ({elapsed:.1f}s)")
    result = {
        "index": index, "target_id": target_id,
        "entity_name": entity_name, "status": "success",
        "model": model, "temperature": temperature,
        "elapsed_seconds": round(elapsed, 1),
        "analysis_method": "graphrag",
        "analysis": _prettify_json_response(analysis).split("\n") if isinstance(analysis, str) else analysis,
        "formatted_subgraph": formatted_subgraph,
        "graphrag_meta": {
            "l1_community_count": len(analyzer.communities),
            "l2_super_community_count": len(analyzer.super_communities),
            "l1_summaries": analyzer.l1_summaries,
            "l2_summaries": analyzer.l2_summaries,
        },
    }
    if english_subgraph:
        result["english_subgraph"] = english_subgraph
    return result
