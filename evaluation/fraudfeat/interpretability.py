# -*- coding: utf-8 -*-
"""Blind LLM interpretability scoring with an English rubric and domain brief.

The judge sees only the feature name, computation logic, and optional business
explanation. Labels, IV values, method names, and window numbers are never included
in model-facing messages. The domain brief is generated exclusively from an
explicitly supplied public schema, or loaded from a frozen text file.

Only the first score per entry and judge contributes to headline and top-k means.
Additional repeats are used exclusively for stability diagnostics. Entries are
identified by ``feature_name#pool_index`` so duplicate names remain independent.

``--judges mock`` is a deterministic, fully offline backend intended only for
pipeline tests. It never opens a network connection and is not a semantic judge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .pool import (
    analysis_pool, entry_key, feature_iv, load_features, load_iv_jsonl,
    prepare_features, ranked_indices,
)


__all__ = [
    "RUBRIC_PROMPT", "JudgeBackend", "MockJudgeBackend", "build_domain_brief",
    "feature_card", "build_requests", "score_requests", "summarize",
    "render_report", "main",
]

logger = logging.getLogger("fraudfeat.interpretability")
DEFAULT_MODEL = os.environ.get("FRAUDFEAT_JUDGE_MODEL", "gpt-4o-mini")
DEFAULT_BASE_URL = os.environ.get("FRAUDFEAT_JUDGE_BASE_URL", "https://api.openai.com/v1")
DEFAULT_API_KEY = os.environ.get("FRAUDFEAT_JUDGE_API_KEY", "")
_TOPK = (10, 30, 50, 100)
RUBRIC_PROMPT = """You are a risk-control domain expert. Score whether the feature name, computation logic, and business explanation are interpretable and consistent with business intuition.
Use four checks without subscores: A real business meaning; B consistency between logic and explanation; C a clear risk mechanism consistent with domain roles; D an actionable strategy use.
Integer scale: 1 means no real meaning or contradiction; 2 means readable but weak on B/C/D; 3 means A+B hold but C or D is unclear; 4 means A+B+C hold and D is broadly actionable; 5 means all four are precise and directly usable.
Cap at 2 for explicit logic contradictions or generic boilerplate. Cap at 3 for merely restating logic or adding unsupported business claims. Judge interpretability only, never IV or feature strength.
Return JSON only: {"score":1-5,"rationale":"one concise reason"}."""

_FAMILIES: Tuple[Tuple[Tuple[str, ...], str, str], ...] = (
    (("wechat_pay_account", "merchant_pay_account"), "Accounts",
     "The evaluated target is a personal WeChat Pay account; the merchant pay account is a universal receiving account"),
    (("individual_customer", "natural_person"), "Subjects",
     "Real-name ownership and customer profiles associated with accounts"),
    (("bank_card", "phone_number"), "Shared media",
     "Phone numbers and bank cards; sharing these across accounts is a strong association signal"),
    (("wechat_pay_bill_deletion_event", "wechat_pay_card_back_information_query_event"),
     "Individual payment behavior events",
     "Bill deletion and card-back information queries are weak alone and become meaningful in unusual combinations or sequences"),
)


def _fmt_count(value: Any) -> str:
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        return str(value)
    if number >= 1e6:
        return f"{number / 1e6:.1f}M"
    if number >= 1e3:
        return f"{number / 1e3:.1f}K"
    return str(int(number))


def _family_of(entity_type: str) -> str:
    for members, name, _note in _FAMILIES:
        if str(entity_type) in members:
            return name
    return "Other"


def _display_entity(entity: Dict[str, Any]) -> str:
    entity_type = str(entity.get("entity_type") or "")
    description = str(entity.get("description") or "").strip()
    return entity_type + (f": {description}" if description else "")


def _relation_name(relation: Dict[str, Any]) -> str:
    return str(relation.get("en") or "").strip() or "unnamed relation"


def build_domain_brief(schema: Dict[str, Any]) -> str:
    """Render an English, method-independent brief from public schema facts only."""
    entities = [e for e in (schema.get("entities") or []) if isinstance(e, dict)]
    relations = [r for r in (schema.get("relations") or []) if isinstance(r, dict)]
    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for entity in entities:
        by_family.setdefault(_family_of(str(entity.get("entity_type") or "")), []).append(entity)
    n_trade = sum(1 for relation in relations if relation.get("is_trade"))
    n_nontrade = len(relations) - n_trade
    lines = [
        "[DOMAIN BRIEF — METHOD-INDEPENDENT GRAPH FACTS]",
        "",
        "1. Scenario",
        f"- The graph models WeChat Pay risk control with {len(entities)} entity types "
        f"and {len(relations)} relation types. Nodes represent accounts, subjects, "
        "shared media, or behavior events; edges represent transfers, links, or events.",
        "- Risk targets are personal WeChat Pay accounts marked as fraud.",
        "- A feature computes a structural aggregate along a relation chain around one target account.",
        "- Vertex properties are compared as strings. Enumerated values are defined by "
        "schema fields[].values and must be interpreted through that dictionary.",
        "",
        f"2. Entity types ({len(entities)})",
    ]
    ordered_families = [name for _prefixes, name, _note in _FAMILIES] + ["Other"]
    for family in ordered_families:
        members = by_family.get(family) or []
        if not members:
            continue
        note = next((note for _p, name, note in _FAMILIES if name == family), "")
        lines.append(f"- {family} ({len(members)}): {note}.")
        for entity in sorted(members, key=lambda item: str(item.get("entity_type") or "")):
            lines.append(f"  - {_display_entity(entity)}")
            for field in entity.get("fields") or []:
                name = str(field.get("name") or "?")
                dtype = str(field.get("dtype") or "")
                description = str(field.get("description") or "").strip()
                values = field.get("values") or {}
                enum_text = "; ".join(f"{key}={value}" for key, value in values.items())
                details = "; ".join(text for text in (description, enum_text) if text)
                type_text = f" ({dtype})" if dtype else ""
                lines.append(f"    - {name}{type_text}" + (f": {details}" if details else ""))
    edge_fields = schema.get("transaction_edge_fields") or []
    if edge_fields:
        lines += ["", "Common transaction-edge fields:"]
        for field in edge_fields:
            name = str(field.get("name") or "?")
            dtype = str(field.get("dtype") or "")
            description = str(field.get("description") or "").strip()
            values = field.get("values") or {}
            enum_text = "; ".join(f"{key}={value}" for key, value in values.items())
            details = "; ".join(text for text in (description, enum_text) if text)
            type_text = f" ({dtype})" if dtype else ""
            lines.append(f"  - {name}{type_text}" + (f": {details}" if details else ""))
    has_counts = any(relation.get("rows") is not None for relation in relations)
    lines += ["", f"3. Relations ({len(relations)}: {n_trade} transaction + {n_nontrade} non-transaction)"]
    if has_counts:
        lines.append("Edge counts describe released-data scale; very sparse relations provide fragile statistical evidence.")
    lines.append("Transaction relations:")

    def edge_suffix(relation: Dict[str, Any]) -> str:
        if relation.get("rows") is None:
            return ""
        return f"; {_fmt_count(relation.get('rows'))} edges"

    for relation in relations:
        if not relation.get("is_trade"):
            continue
        for endpoint in relation.get("join_fields") or [{}]:
            src, dst = str(endpoint.get("src") or "?"), str(endpoint.get("dst") or "?")
            lines.append(f"- {_relation_name(relation)} ({src} -> {dst}{edge_suffix(relation)})")
    lines.append("Non-transaction relations (single stored direction):")
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for relation in relations:
        if relation.get("is_trade"):
            continue
        for endpoint in relation.get("join_fields") or [{}]:
            key = (str(endpoint.get("src") or "?"), str(endpoint.get("dst") or "?"))
            groups.setdefault(key, []).append({"relation": relation})
    for (src, dst), items in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        has_all_counts = all(item["relation"].get("rows") is not None for item in items)
        total = sum(int(item["relation"].get("rows") or 0) for item in items)
        scope = f"{len(items)} types" + (f", {_fmt_count(total)} edges" if has_all_counts else "")
        detail_items = []
        for item in items[:6]:
            relation = item["relation"]
            text = _relation_name(relation)
            if relation.get("rows") is not None:
                text += f" ({_fmt_count(relation.get('rows'))} edges)"
            detail_items.append(text)
        more = f", and {len(items) - 6} more" if len(items) > 6 else ""
        lines.append(f"- {src} -> {dst} ({scope}): {', '.join(detail_items)}{more}")
    if any(str(relation.get("description") or "").strip() for relation in relations):
        lines += ["", "Relation semantics:"]
        for relation in relations:
            description = str(relation.get("description") or "").strip()
            if description:
                lines.append(f"- {_relation_name(relation)}: {description}")
    lines += [
        "",
        "4. Expected value types",
        "- count: deduplicated number of objects; flag: 1/0 existence.",
        "- amount: sum, mean, minimum, or maximum of a numeric edge attribute.",
        "- ratio: division of two explicitly defined counts, with numerator and denominator stated.",
        "",
        "5. Labels and validity",
        "- Labels cover personal WeChat Pay accounts only (fraud=1, non-fraud=0); "
        "merchant accounts are unlabeled.",
        "- Only entities and relations listed in this brief exist in the released graph. "
        "References to absent label-derived, intervention, or compliance entities are invalid and receive score 1.",
        "- Edge amounts are stored in fen. Describing their business interpretation in yuan is valid.",
    ]
    return "\n".join(lines)


def system_prompt(brief: str = "") -> str:
    return RUBRIC_PROMPT + (("\n\n" + brief) if brief.strip() else "")


def rubric_sha1(brief: str = "") -> str:
    return hashlib.sha1(system_prompt(brief).encode("utf-8")).hexdigest()[:12]


def domain_sha1(brief: str = "") -> str:
    return hashlib.sha1((brief or "").encode("utf-8")).hexdigest()[:12]


def _load_features(path: str | Path) -> List[Dict[str, Any]]:
    return load_features(path)


def _load_iv_ranking(path: Optional[str]) -> Dict[str, float]:
    return load_iv_jsonl(path)


def feature_card(feature: Dict[str, Any], *, with_explanation: bool = True) -> Dict[str, Any]:
    explanation = str(feature.get("explanation") or "") if with_explanation else ""
    source = str(feature.get("explanation_source") or "business_explanation")
    return {
        "feature_name": feature.get("feature_name") or "",
        "logic": feature.get("logic") or "",
        "explanation": explanation,
        "explanation_source": source if explanation else "logic_only",
    }


def build_requests(features: List[Dict[str, Any]], *, judges: List[str], brief: str,
                   repeat: int = 1, with_explanation: bool = True,
                   stability_features: int = 0, stability_repeat: int = 3,
                   seed: int = 20260817) -> List[Dict[str, Any]]:
    """Build one request per pool entry/judge, plus deterministic stability repeats."""
    features = prepare_features(features)
    repeated_indices: set[int] = set()
    if stability_features > 0 and features:
        repeated_indices = set(random.Random(seed).sample(
            list(range(len(features))), min(stability_features, len(features))))
    fingerprint = rubric_sha1(brief)
    requests: List[Dict[str, Any]] = []
    for position, feature in enumerate(features):
        name = str(feature.get("feature_name") or "")
        if not name:
            continue
        card = feature_card(feature, with_explanation=with_explanation)
        card_sha1 = hashlib.sha1(json.dumps(card, ensure_ascii=False, sort_keys=True)
                                 .encode("utf-8")).hexdigest()[:12]
        pool_index = int(feature["pool_index"])
        count = max(1, stability_repeat) if position in repeated_indices else max(1, repeat)
        for judge in judges:
            for rep in range(count):
                requests.append({
                    "request_id": f"{pool_index}|{name}|{judge}|{rep}|{card_sha1}",
                    "card_sha1": card_sha1,
                    "index": position,
                    "pool_index": pool_index,
                    "entry_key": entry_key(feature),
                    "judge": judge,
                    "repeat": rep,
                    "rubric_sha1": fingerprint,
                    **card,
                })
    return requests


def _parse_judge_json(content: str) -> Optional[Dict[str, Any]]:
    text = (content or "").strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines()
                         if not line.strip().startswith("```"))
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            return None
        depth, end = 0, -1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end < 0:
            return None
        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    try:
        score = int(data.get("score"))
    except (TypeError, ValueError):
        return None
    if not 1 <= score <= 5:
        return None
    return {"score": score, "rationale": str(data.get("rationale") or "").strip()}


def _post_json(url: str, payload: Dict[str, Any], api_key: str,
               timeout: float) -> Tuple[Optional[Dict[str, Any]], str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8")), ""
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            detail = ""
        return None, f"HTTP {exc.code}: {detail}"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _judge_user_prompt(card: Dict[str, Any]) -> str:
    name = str(card.get("feature_name") or "").strip()
    logic = str(card.get("logic") or "None").strip()
    explanation = str(card.get("explanation") or "None").strip()
    return (f"Feature name: {name}\nComputation logic: {logic}\n"
            f"Business explanation: {explanation}")


class JudgeBackend:
    """OpenAI-compatible chat/completions judge."""

    def __init__(self, base_url: str, api_key: str, model: str, *, brief: str = "",
                 timeout: float = 300.0, max_retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.brief = brief
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def url(self) -> str:
        return (self.base_url if self.base_url.endswith("/chat/completions")
                else self.base_url + "/chat/completions")

    def score(self, card: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt(self.brief)},
                         {"role": "user", "content": _judge_user_prompt(card)}],
            "stream": False,
            "temperature": 0,
            "max_tokens": 256,
        }
        last_error = "Not sent"
        for attempt in range(1 + self.max_retries):
            response, error = _post_json(self.url, payload, self.api_key, self.timeout)
            if response is None:
                for key in ("temperature", "max_tokens"):
                    if key in payload and key in (error or "").lower():
                        payload.pop(key, None)
                last_error = error
            else:
                try:
                    content = response["choices"][0]["message"]["content"]
                except Exception as exc:  # noqa: BLE001
                    last_error = f"Response parsing failed: {exc}"
                    content = ""
                if content:
                    parsed = _parse_judge_json(content)
                    if parsed:
                        return parsed
                    last_error = f"Expected a JSON score from 1 to 5: {content}"
            if attempt < self.max_retries:
                time.sleep(1.5 * (attempt + 1))
        return {"score": None, "rationale": last_error}

    def probe(self) -> bool:
        result = self.score({"feature_name": "connectivity_probe", "logic": "-",
                             "explanation": "-"})
        ok = result.get("score") is not None
        logger.info("[probe] %s -> %s", self.model,
                    f"available score={result.get('score')}" if ok
                    else f"unavailable: {result.get('rationale')}")
        return ok


class MockJudgeBackend:
    """Deterministic offline stub; its output is not an interpretability assessment."""

    def __init__(self, model: str = "mock", *, brief: str = ""):
        self.model = model
        self.brief = brief

    def score(self, card: Dict[str, Any]) -> Dict[str, Any]:
        text = (self.model + "\n" + system_prompt(self.brief) + "\n"
                + _judge_user_prompt(card)).encode("utf-8")
        score = 1 + hashlib.sha256(text).digest()[0] % 5
        return {"score": score,
                "rationale": "Deterministic offline mock result; not a semantic assessment."}

    def probe(self) -> bool:
        return True


def _make_backend(name: str, base_url: str, api_key: str, brief: str,
                  timeout: float) -> Any:
    if name.strip().lower() == "mock":
        return MockJudgeBackend(name, brief=brief)
    return JudgeBackend(base_url, api_key, name, brief=brief, timeout=timeout)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def score_requests(requests: List[Dict[str, Any]], out_dir: Path, *,
                   base_url: str, api_key: str, brief: str,
                   workers: int = 1, timeout: float = 300.0,
                   force: bool = False) -> Dict[str, int]:
    """Score requests idempotently; a changed rubric is never treated as completed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "judge_scores.jsonl"
    previous = _read_jsonl(scores_path)
    by_id = {str(row.get("request_id")): row for row in requests}
    done = set()
    if not force:
        for row in previous:
            request_id = str(row.get("request_id") or "")
            request = by_id.get(request_id)
            if (request and isinstance(row.get("score"), int)
                    and row.get("rubric_sha1") == request.get("rubric_sha1")
                    and row.get("card_sha1") == request.get("card_sha1")):
                done.add(request_id)
    todo = [row for row in requests if str(row.get("request_id")) not in done]
    stats = {"scored": 0, "skipped": len(requests) - len(todo), "failed": 0}
    backends: Dict[str, Any] = {}

    def backend(name: str) -> Any:
        if name not in backends:
            backends[name] = _make_backend(name, base_url, api_key, brief, timeout)
        return backends[name]

    def run_one(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        result = backend(str(request["judge"])).score(request)
        if result.get("score") is None:
            stats["failed"] += 1
            logger.warning("[judge] %s failed: %s", request.get("request_id"),
                           str(result.get("rationale"))[:300])
            return None
        return {
            "request_id": request["request_id"],
            "card_sha1": request.get("card_sha1"),
            "index": request.get("index"),
            "pool_index": request.get("pool_index"),
            "entry_key": request.get("entry_key"),
            "feature_name": request.get("feature_name"),
            "logic": request.get("logic") or "",
            "judge": request.get("judge"),
            "repeat": request.get("repeat", 0),
            "score": int(result["score"]),
            "rationale": result.get("rationale") or "",
            "rubric_sha1": request.get("rubric_sha1"),
            "explanation_source": request.get("explanation_source"),
        }

    rows: List[Dict[str, Any]] = []
    if workers <= 1:
        for request in todo:
            row = run_one(request)
            if row:
                rows.append(row)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for row in executor.map(run_one, todo):
                if row:
                    rows.append(row)
    stats["scored"] = len(rows)
    if rows:
        todo_ids = {str(row.get("request_id")) for row in todo}
        with scores_path.open("w" if force else "a", encoding="utf-8") as handle:
            if force:
                for row in previous:
                    if str(row.get("request_id")) not in todo_ids:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return stats


def _ranks(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def _spearman(a: List[float], b: List[float]) -> Optional[float]:
    if len(a) < 3 or len(a) != len(b):
        return None
    ra, rb = _ranks(a), _ranks(b)
    mean_a, mean_b = sum(ra) / len(ra), sum(rb) / len(rb)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    denominator_a = sum((x - mean_a) ** 2 for x in ra) ** 0.5
    denominator_b = sum((y - mean_b) ** 2 for y in rb) ** 0.5
    return (round(numerator / (denominator_a * denominator_b), 6)
            if denominator_a and denominator_b else None)


def _kendalls_w(judge_rows: List[List[float]]) -> Optional[float]:
    if len(judge_rows) < 2:
        return None
    m = len(judge_rows)
    n = len(judge_rows[0])
    if n < 2 or any(len(row) != n for row in judge_rows):
        return None
    rank_rows = [_ranks(row) for row in judge_rows]
    rank_sums = [sum(rank_rows[j][i] for j in range(m)) for i in range(n)]
    mean_rank = m * (n + 1) / 2
    deviation = sum((value - mean_rank) ** 2 for value in rank_sums)
    tie_total = 0.0
    for row in judge_rows:
        counts: Dict[float, int] = {}
        for value in row:
            counts[value] = counts.get(value, 0) + 1
        tie_total += sum(count ** 3 - count for count in counts.values() if count > 1)
    denominator = m * m * (n ** 3 - n) - m * tie_total
    return round(12 * deviation / denominator, 6) if denominator > 0 else None


def _mean(values: Iterable[float], digits: int = 4) -> Optional[float]:
    values = list(values)
    return round(sum(values) / len(values), digits) if values else None


def _valid_score(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = int(value)
    return score if score == value and 1 <= score <= 5 else None


def _score_entry_key(row: Dict[str, Any], pool_keys: set[str],
                     by_name: Dict[str, List[str]]) -> Optional[str]:
    explicit = str(row.get("entry_key") or "")
    if explicit in pool_keys:
        return explicit
    if isinstance(row.get("pool_index"), int) and not isinstance(row.get("pool_index"), bool):
        key = entry_key(row)
        if key in pool_keys:
            return key
    name = str(row.get("feature_name") or "")
    matches = by_name.get(name) or []
    # Name-only legacy scores are safe only for an unambiguous pool entry.
    if not explicit and row.get("pool_index") is None and len(matches) == 1:
        return matches[0]
    return None


def summarize(features: List[Dict[str, Any]], scores: List[Dict[str, Any]], *,
              pool_limit: int = 0, iv: Optional[Mapping[str, float]] = None,
              rubric_sha: Optional[str] = None) -> Dict[str, Any]:
    """Aggregate first scores within the fixed input-prefix analysis pool."""
    all_features = prepare_features(features)
    pool = analysis_pool(all_features, pool_limit)
    pool_keys = {entry_key(feature) for feature in pool}
    by_name: Dict[str, List[str]] = {}
    for feature in pool:
        by_name.setdefault(str(feature.get("feature_name") or ""), []).append(entry_key(feature))
    iv_map = iv or {}
    groups: Dict[str, Dict[str, List[Tuple[int, int, int]]]] = {}
    in_pool_rows = 0
    failed = 0
    excluded = 0
    for order, row in enumerate(scores):
        if rubric_sha is not None and row.get("rubric_sha1") != rubric_sha:
            excluded += 1
            continue
        key = _score_entry_key(row, pool_keys, by_name)
        if key is None:
            excluded += 1
            continue
        score = _valid_score(row.get("score"))
        if score is None:
            failed += 1
            continue
        judge = str(row.get("judge") or "")
        if not judge:
            excluded += 1
            continue
        repeat = row.get("repeat")
        repeat = int(repeat) if isinstance(repeat, (int, float)) else 0
        groups.setdefault(key, {}).setdefault(judge, []).append((repeat, order, score))
        in_pool_rows += 1
    first: Dict[str, Dict[str, int]] = {}
    for key, per_judge in groups.items():
        first[key] = {}
        for judge, observations in per_judge.items():
            observations.sort(key=lambda item: (item[0], item[1]))
            first[key][judge] = observations[0][2]
    feature_means = {key: sum(per.values()) / len(per)
                     for key, per in first.items() if per}
    scored_features = [feature for feature in pool if entry_key(feature) in feature_means]
    judge_names = sorted({judge for per in first.values() for judge in per})
    judge_means: Dict[str, float] = {}
    judge_stats: Dict[str, Any] = {}
    for judge in judge_names:
        values = [first[entry_key(feature)][judge] for feature in pool
                  if judge in first.get(entry_key(feature), {})]
        if values:
            judge_means[judge] = sum(values) / len(values)
        judge_stats[judge] = {"n": len(values), "mean": _mean(values)}
    summary: Dict[str, Any] = {
        "features_in": len(all_features),
        "features_scored": len(scored_features),
        "pool_size": len(pool),
        "pool_limit": pool_limit,
        "coverage": _mean([len(scored_features) / len(pool)]) if pool else None,
        "scores": in_pool_rows,
        "main_scores": sum(len(per) for per in first.values()),
        "failed": failed,
        "excluded_scores": excluded,
        "overall_mean": _mean(judge_means.values()),
        "judges": judge_stats,
    }
    order = ranked_indices(pool, iv_map)
    for k in _TOPK:
        values = [feature_means[entry_key(pool[i])] for i in order[:k]
                  if entry_key(pool[i]) in feature_means]
        summary[f"top{k}_mean_score"] = _mean(values)
        summary[f"top{k}_n"] = len(values)

    stability_groups = [(key, judge, [item[2] for item in sorted(observations)])
                        for key, per in groups.items() for judge, observations in per.items()
                        if len(observations) > 1]
    if stability_groups:
        standard_deviations: List[float] = []
        spans: List[int] = []
        for _key, _judge, values in stability_groups:
            mean = sum(values) / len(values)
            standard_deviations.append((sum((value - mean) ** 2 for value in values)
                                        / len(values)) ** 0.5)
            spans.append(max(values) - min(values))
        summary["stability"] = {
            "groups": len(stability_groups),
            "mean_std": _mean(standard_deviations),
            "max_span": max(spans),
            "exact_agreement": round(sum(1 for _k, _j, values in stability_groups
                                         if len(set(values)) == 1)
                                     / len(stability_groups), 4),
        }
    agreements = 0
    comparisons = 0
    pair_correlations: Dict[str, Optional[float]] = {}
    for i, judge_a in enumerate(judge_names):
        for judge_b in judge_names[i + 1:]:
            values_a: List[float] = []
            values_b: List[float] = []
            for key, per in first.items():
                if judge_a in per and judge_b in per:
                    values_a.append(float(per[judge_a]))
                    values_b.append(float(per[judge_b]))
                    comparisons += 1
                    agreements += int(per[judge_a] == per[judge_b])
            pair_correlations[f"{judge_a}~{judge_b}"] = _spearman(values_a, values_b)
    summary["judge_agreement"] = round(agreements / comparisons, 4) if comparisons else None
    summary["judge_pairs"] = comparisons
    summary["judge_pair_spearman"] = pair_correlations
    if len(judge_names) == 2:
        summary["judge_spearman"] = pair_correlations.get(f"{judge_names[0]}~{judge_names[1]}")
    complete = [key for key, per in first.items() if all(j in per for j in judge_names)]
    matrix = [[float(first[key][judge]) for key in complete] for judge in judge_names]
    summary["judge_kendall_w"] = _kendalls_w(matrix) if matrix else None
    summary["judge_kendall_n"] = len(complete)
    summary["feature_means"] = []
    for i in order[:100]:
        feature = pool[i]
        key = entry_key(feature)
        if key not in feature_means:
            continue
        summary["feature_means"].append({
            "feature_name": feature.get("feature_name"),
            "pool_index": feature.get("pool_index"),
            "entry_key": key,
            "window": feature.get("window"),
            "n": len(first[key]),
            "mean_score": round(feature_means[key], 4),
            "iv": feature_iv(feature, iv_map),
        })
    return summary


def render_report(summary: Dict[str, Any], *, features_path: str, brief_sha: str,
                  rubric_sha: str, judges: List[str]) -> str:
    lines = [
        "FraudFeat interpretability report (1-5 blind judging)",
        "=" * 64,
        f"Features: {features_path}",
        f"Judges: {', '.join(judges)}",
        f"rubric_sha1: {rubric_sha} (full system message)",
        f"domain_sha1: {brief_sha}",
        f"Scored entries: {summary['features_scored']} / {summary['pool_size']} "
        f"(input {summary['features_in']})",
        f"Score rows: {summary['scores']} (headline {summary['main_scores']}; "
        f"failed {summary['failed']}; excluded {summary['excluded_scores']})",
        f"Overall mean (equal weight per judge): {summary['overall_mean']}",
        "",
        "Per-judge first-score means:",
    ]
    for judge, stats in summary["judges"].items():
        lines.append(f"  {judge:<32} n={stats['n']:<6} mean={stats['mean']}")
    lines.append("Top-k means (IV-ranked within the fixed analysis pool):")
    for k in _TOPK:
        lines.append(f"  top{k:<4}: {summary.get(f'top{k}_mean_score')} "
                     f"n={summary.get(f'top{k}_n')}")
    if summary.get("judge_agreement") is not None:
        lines.append(f"Pairwise exact agreement: {summary['judge_agreement']} "
                     f"({summary['judge_pairs']} comparisons)")
    if summary.get("judge_kendall_w") is not None:
        lines.append(f"Kendall's W: {summary['judge_kendall_w']} "
                     f"(n={summary['judge_kendall_n']})")
    if summary.get("stability"):
        stability = summary["stability"]
        lines.append(f"Stability: groups={stability['groups']}, "
                     f"mean_std={stability['mean_std']}, max_span={stability['max_span']}, "
                     f"exact_agreement={stability['exact_agreement']}")
    lines += ["", "Highest-IV scored entries:"]
    for i, row in enumerate(summary["feature_means"][:20], 1):
        lines.append(f"  {i:>2}. {row['entry_key']} score={row['mean_score']} "
                     f"judges={row['n']} iv={row['iv']}")
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fraudfeat.interpretability",
        description="Blind interpretability scoring with an English rubric and public-schema brief")
    parser.add_argument("--features", help="Feature JSON/JSONL file")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--schema", help="Public schema.json or its data directory")
    parser.add_argument("--brief", help="Frozen English domain brief (takes precedence over --schema)")
    parser.add_argument("--save-brief", help="Write the generated brief and exit when --features is absent")
    parser.add_argument("--judges", default=DEFAULT_MODEL,
                        help="Comma-separated model names, or 'mock' for fully offline tests")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key; prefer FRAUDFEAT_JUDGE_API_KEY")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat count per entry; only repeat 0 contributes to headline scores")
    parser.add_argument("--stability-features", type=int, default=0)
    parser.add_argument("--stability-repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260817, help="Stability sampling seed")
    parser.add_argument("--pool-limit", type=int, default=100,
                        help="Take the input prefix of N entries before IV ranking; 0=all")
    parser.add_argument("--iv-jsonl", help="effectiveness.jsonl for pool-local top-k ranking")
    parser.add_argument("--no-explanation", action="store_true",
                        help="Judge feature name and computation logic only")
    parser.add_argument("--rescore", action="store_true", help="Replace scores covered by this run")
    parser.add_argument("--probe", action="store_true", help="Probe the first judge and exit")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    judges = [item.strip() for item in str(args.judges).split(",") if item.strip()]
    if not judges:
        logger.error("At least one judge is required")
        return 2
    brief = ""
    if args.brief:
        brief = Path(args.brief).read_text(encoding="utf-8")
    elif args.schema:
        from .graph_store import load_schema
        schema_path = Path(args.schema)
        data_dir = (schema_path.parent if schema_path.is_file()
                    or schema_path.suffix.lower() == ".json" else schema_path)
        brief = build_domain_brief(load_schema(data_dir))
    if args.save_brief:
        if not brief.strip():
            logger.error("--save-brief requires --schema or --brief with public relation facts")
            return 2
        save_path = Path(args.save_brief)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(brief + "\n", encoding="utf-8")
        logger.info("Domain brief written to %s (%d characters)", save_path, len(brief))
        if not args.features:
            return 0
    if args.probe:
        backend = _make_backend(judges[0], args.base_url, args.api_key, brief, args.timeout)
        return 0 if backend.probe() else 1
    if not args.features:
        logger.error("--features is required unless using --save-brief without scoring or --probe")
        return 2
    if not brief:
        logger.warning("No --brief or --schema supplied; the judge receives only the rubric")
    all_features = _load_features(args.features)
    features = analysis_pool(all_features, args.pool_limit)
    if not features:
        logger.error("No features to score: %s", args.features)
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    requests = build_requests(
        features, judges=judges, brief=brief, repeat=args.repeat,
        with_explanation=not args.no_explanation,
        stability_features=args.stability_features,
        stability_repeat=args.stability_repeat, seed=args.seed)
    request_path = out_dir / "judge_requests.jsonl"
    request_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                    for row in requests), encoding="utf-8")
    logger.info("Requests: %d (%d/%d pool entries x %d judges) -> %s",
                len(requests), len(features), len(all_features), len(judges), request_path)
    stats = score_requests(requests, out_dir, base_url=args.base_url,
                           api_key=args.api_key, brief=brief, workers=args.workers,
                           timeout=args.timeout, force=args.rescore)
    logger.info("Scoring: new=%d skipped=%d failed=%d",
                stats["scored"], stats["skipped"], stats["failed"])
    scores = _read_jsonl(out_dir / "judge_scores.jsonl")
    iv = _load_iv_ranking(args.iv_jsonl)
    fingerprint = rubric_sha1(brief)
    summary = summarize(all_features, scores, pool_limit=args.pool_limit, iv=iv,
                        rubric_sha=fingerprint)
    summary["rubric_sha1"] = fingerprint
    summary["domain_sha1"] = domain_sha1(brief)
    summary["judge_models"] = judges
    report = render_report(summary, features_path=str(args.features),
                           brief_sha=domain_sha1(brief), rubric_sha=fingerprint,
                           judges=judges)
    (out_dir / "interpretability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "interpretability_report.txt").write_text(report, encoding="utf-8")
    if not args.quiet:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
