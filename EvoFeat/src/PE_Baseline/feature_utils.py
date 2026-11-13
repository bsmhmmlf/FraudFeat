import json

import math

import os

import re

import time

from collections import deque

from typing import Any, Dict, List, Optional

import requests

from decimal import Decimal


API_URL = os.environ.get("SYSTEM_LLM_API_URL", "<YOUR_LLM_API_URL>")
DEFAULT_API_KEY = os.environ.get("SYSTEM_LLM_API_KEY", "<YOUR_LLM_API_KEY>")
HEADERS = {
    "Authorization": "Bearer {}".format(DEFAULT_API_KEY),
    "Content-Type": "application/json",
}
DEFAULT_MODEL = "hy4-preview"

TARGET_SUBGRAPH_TOKENS = 96_000

ENABLE_PRUNING = True

ENTITY_TYPE_PREFIX = {
    "account": "A",
    "object": "O",
    "operation": "B",
    "operation_behavior": "B",
    "subject": "S",
}
API_TIMEOUT = 600
API_MAX_RETRIES = 5

API_RETRY_BASE = 5

API_RETRY_MAX = 80


def _json_default(o: Any):
    if isinstance(o, Decimal):
        return float(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def estimate_tokens_simple(text: str) -> int:
    if not text:
        return 0
    cjk = 0
    other = 0
    for ch in text:
        o = ord(ch)
        if (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF
                or 0x3000 <= o <= 0x303F or 0xFF00 <= o <= 0xFFEF
                or 0x3040 <= o <= 0x30FF):
            cjk += 1
        else:
            other += 1
    return int(cjk * 2 + other * 0.25)


def estimate_subgraph_tokens(subgraph: Dict[str, Any]) -> int:
    text = json.dumps(subgraph, ensure_ascii=False, indent=2, default=_json_default)
    return estimate_tokens_simple(text)


def _prefix_letter_from_type_str(type_str: str) -> str:
    m = re.search(r"\(([a-zA-Z_]+)", type_str or "")
    if not m:
        return "N"
    name = m.group(1).lower()
    return ENTITY_TYPE_PREFIX.get(name, name[:1].upper())


def _entity_of_node(node: Dict[str, Any]) -> str:
    ent = node.get("entity")
    if ent:
        return ent
    vid = node.get("vid", "")
    return vid


def rebuild_vid2short(
    eng_nodes: List[Dict[str, Any]],
    target_vid: str,
    entity_meta: Dict[str, Dict[str, str]],
) -> Dict[str, str]:
    grouped: Dict[tuple, List[dict]] = {}
    for n in eng_nodes:
        vid = n.get("vid", "")
        if not vid:
            continue
        entity = _entity_of_node(n)
        letter = (entity_meta.get(entity, {}) or {}).get("prefix", "N")
        key = (letter, entity)
        grouped.setdefault(key, []).append({
            "vid": vid,
            "hop": int(n.get("first_hop", 0) or 0),
            "is_target": vid == target_vid,
        })
    group_meta: Dict[tuple, dict] = {}
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


def _target_vid(eng: Dict[str, Any]) -> str:
    for n in eng.get("nodes", []) or []:
        if int(n.get("first_hop", 0) or 0) == 0:
            return n.get("vid", "")
    nodes = eng.get("nodes") or []
    return nodes[0].get("vid", "") if nodes else ""


def _skeleton_vids(
    eng: Dict[str, Any],
    target_vid: str,
    adj: Dict[str, set],
    hop_of: Dict[str, int],
) -> set:
    if not target_vid or target_vid not in adj:
        return set()
    parent: Dict[str, str] = {target_vid: ""}
    dist: Dict[str, int] = {target_vid: 0}
    dq = deque([target_vid])
    while dq:
        u = dq.popleft()
        for w in adj.get(u, ()):
            if w not in dist:
                dist[w] = dist[u] + 1
                parent[w] = u
                dq.append(w)
    if not dist:
        return {target_vid}
    max_layer = max(dist.values())
    layer_nodes: Dict[int, List[str]] = {}
    for v, d in dist.items():
        layer_nodes.setdefault(d, []).append(v)
    degree = {v: len(adj.get(v, ())) for v in dist}
    skeleton: set = {target_vid}
    for layer in range(max_layer, 0, -1):
        cands = layer_nodes.get(layer, [])
        if not cands:
            continue
        rep = max(cands, key=lambda v: degree.get(v, 0))
        cur = rep
        while cur:
            skeleton.add(cur)
            cur = parent.get(cur, "")
    return skeleton


def compute_vid_importance(
    eng: Dict[str, Any],
    target_vid: str,
    rel_rarity: Dict[str, int] = None,
    rare_max: int = 10,
) -> Dict[str, float]:
    nodes = eng.get("nodes", []) or []
    edges = eng.get("edges", []) or []
    entity_of: Dict[str, str] = {n.get("vid", ""): _entity_of_node(n) for n in nodes}
    hop_of: Dict[str, int] = {n.get("vid", ""): int(n.get("first_hop", 0) or 0) for n in nodes}
    adj: Dict[str, set] = {v: set() for v in entity_of}
    degree: Dict[str, int] = {v: 0 for v in entity_of}
    for e in edges:
        s, d = e.get("src", ""), e.get("dst", "")
        if s in adj and d in adj:
            adj[s].add(d)
            adj[d].add(s)
            degree[s] += 1
            degree[d] += 1
    skeleton = _skeleton_vids(eng, target_vid, adj, hop_of)
    rare_protected: set = set()
    if rel_rarity is not None:
        for e in edges:
            key = _normalize_relation_key(e.get("join_field", ""))
            if not key:
                continue
            cnt = rel_rarity.get(key, 0)
            if cnt <= rare_max:
                s, d = e.get("src", ""), e.get("dst", "")
                if s in entity_of:
                    rare_protected.add(s)
                if d in entity_of:
                    rare_protected.add(d)
    scores: Dict[str, float] = {}
    for v, ent in entity_of.items():
        if v == target_vid:
            scores[v] = float("inf")
            continue
        if v in skeleton:
            scores[v] = float("inf")
            continue
        if v in rare_protected:
            scores[v] = float("inf")
            continue
        deg = degree.get(v, 0)
        base = 1.0
        if deg >= 3:
            base += 100.0 + 10.0 * math.log(deg)
        else:
            base += 10.0 / math.log(deg + math.e)
        scores[v] = base
    return scores


def _normalize_relation_key(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return ""
    parts = []
    for seg in raw.split("|"):
        subs = [s.strip() for s in seg.split("&") if s.strip()]
        parts.append("&".join(sorted(subs)) if subs else seg.strip())
    return "|".join(parts)


def subgraph_richness_stats(subgraph: Dict[str, Any]) -> Dict[str, Any]:
    rel_types: set = set()
    entity_types: set = set()
    n_nodes = 0
    n_edges = 0
    max_hop = 0
    eng = subgraph.get("english_subgraph") or {}
    nodes = eng.get("nodes") or []
    edges = eng.get("edges") or []
    if nodes or edges:
        n_nodes = len(nodes)
        n_edges = len(edges)
        for nd in nodes:
            if isinstance(nd, dict):
                ent = nd.get("entity")
                if ent:
                    entity_types.add(ent)
                max_hop = max(max_hop, int(nd.get("first_hop", 0) or 0))
        for ed in edges:
            if isinstance(ed, dict):
                key = _normalize_relation_key(ed.get("join_field", ""))
                if key:
                    rel_types.add(key)
                max_hop = max(max_hop, int(ed.get("first_hop", 0) or 0))
    else:
        for grp in subgraph.get("node_summary", []) or []:
            if isinstance(grp, dict):
                ename = grp.get("entity") or grp.get("entity_display_name")
                if ename:
                    entity_types.add(ename)
                n_nodes += int(grp.get("count", 0) or 0)
        for item in subgraph.get("relations", []) or []:
            if not isinstance(item, dict):
                continue
            for _pair, labels in item.items():
                for lab in (labels or []):
                    n_edges += 1
                    key = _normalize_relation_key(lab)
                    if key:
                        rel_types.add(key)
    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "max_hop": max_hop,
        "n_relation_types": len(rel_types),
        "n_entity_types": len(entity_types),
        "rel_types": rel_types,
        "entity_types": entity_types,
    }


def call_api(system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL,
             max_retries: int = None, temperature: float = 0.9,
             timeout: int = None) -> Optional[str]:
    if max_retries is None:
        max_retries = API_MAX_RETRIES
    max_retries = max(1, int(max_retries))
    if timeout is None:
        timeout = API_TIMEOUT
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_completion_tokens": 81920,
        "reasoning_effort": "high",
    }

    def _strip_unsupported_params(payload: dict, err_text: str) -> list:
        lowered = (err_text or "").lower()
        removed = []
        for key in ("temperature", "reasoning_effort", "top_p"):
            if key in payload and key in lowered:
                payload.pop(key, None)
                removed.append(key)
        if not removed and "temperature" in payload and any(
            kw in lowered for kw in ("not support", "unsupported", "invalid", "not allowed")
        ):
            payload.pop("temperature", None)
            removed.append("temperature")
        return removed

    def _backoff(attempt: int) -> float:
        return min(API_RETRY_BASE * (2 ** (attempt - 1)), API_RETRY_MAX)

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(API_URL, headers=HEADERS,
                                     data=json.dumps(data), timeout=timeout)
            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
            elif response.status_code == 429 or response.status_code >= 500:
                if attempt < max_retries:
                    wait = _backoff(attempt)
                    tag = "RATE LIMIT" if response.status_code == 429 else "SERVER"
                    print(f"    [{tag} {response.status_code}] Retrying in {wait:.0f}s "
                          f"(attempt {attempt}/{max_retries}) ...")
                    time.sleep(wait)
                    continue
                print(f"    [ERROR] API returned {response.status_code}; maximum retry count reached")
                return None
            else:
                err_text = ""
                try:
                    err_text = response.text or ""
                except Exception:
                    err_text = ""
                stripped = _strip_unsupported_params(data, err_text)
                if stripped and attempt < max_retries:
                    print(f"    [PARAM] API returned {response.status_code}; removed unsupported parameters "
                          f"{stripped}, retrying immediately (attempt {attempt}/{max_retries}) ...")
                    continue
                print(f"    [ERROR] API returned {response.status_code}: {err_text[:200]}")
                if attempt < max_retries:
                    time.sleep(_backoff(attempt))
                    continue
                return None
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                wait = _backoff(attempt)
                print(f"    [TIMEOUT] Request exceeded {timeout}s; retrying in {wait:.0f}s "
                      f"(attempt {attempt}/{max_retries}) ...")
                time.sleep(wait)
                continue
            print(f"    [TIMEOUT] Request exceeded {timeout}s; maximum retry count reached, giving up")
            return None
        except Exception as e:
            if attempt < max_retries:
                wait = _backoff(attempt)
                print(f"    [ERROR] API request failed: {str(e)[:120]}; retrying in {wait:.0f}s "
                      f"(attempt {attempt}/{max_retries}) ...")
                time.sleep(wait)
                continue
            print(f"    [ERROR] API request failed: {str(e)[:120]}; maximum retry count reached")
            return None
    return None


def parse_model_response(response: str):
    if not response or not response.strip():
        return None
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None
