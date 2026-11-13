"""Shared analysis-pool selection, stable entry identity, and IV ranking."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


TOPK = (10, 30, 50, 100)


def _plan_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def prepare_features(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize supported feature fields without discarding submission metadata.

    ``pool_index`` is the stable public identity supplied by an exported pool. If it
    is absent, the zero-based input position is used. Duplicate explicit indices are
    rejected because they would make scores and cached vectors ambiguous.
    """
    out: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for pos, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        f = dict(item)
        raw_index = f.get("pool_index")
        pool_index = (raw_index if isinstance(raw_index, int)
                      and not isinstance(raw_index, bool) else pos)
        if pool_index in seen:
            raise ValueError(f"Duplicate pool_index: {pool_index}")
        seen.add(pool_index)
        name = str(f.get("feature_name") or f.get("name") or f"feature_{pos}")
        plan = _plan_dict(f.get("gremlin_plan") or f.get("plan")
                          or f.get("dsl_program"))
        explanation = str(f.get("business_explanation") or f.get("explanation")
                          or f.get("rationale") or "")
        explanation_source = str(f.get("explanation_source") or (
            "business_explanation" if f.get("business_explanation")
            else "explanation" if f.get("explanation")
            else "rationale" if f.get("rationale") else "logic_only"))
        f.pop("display_name", None)
        f.update({
            "feature_name": name,
            "logic": str(f.get("logic") or f.get("feature_understanding") or ""),
            "business_explanation": explanation,
            "explanation": explanation,
            "explanation_source": explanation_source,
            "expected_value_type": str(f.get("expected_value_type")
                                       or f.get("value_type") or ""),
            "gremlin_plan": plan,
            "plan": plan,
            "pool_index": pool_index,
            "input_index": pos,
            "entry_key": f"{name}#{pool_index}",
        })
        out.append(f)
    return out


def load_features(path: str | Path) -> List[Dict[str, Any]]:
    """Load a JSON array, a ``{"features": [...]}`` object, JSONL, or one feature."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".jsonl", ".ndjson"):
        raw: Any = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        obj = json.loads(text)
        raw = obj.get("features") if isinstance(obj, dict) and "features" in obj else obj
        if isinstance(raw, dict):
            raw = [raw]
    if not isinstance(raw, list):
        raise ValueError(f"Feature file must contain an array or a features array: {p}")
    return prepare_features(raw)


def analysis_pool(features: Iterable[Dict[str, Any]], limit: int = 0) -> List[Dict[str, Any]]:
    """Take the input prefix first; never rank by IV before applying this cutoff."""
    pool = list(features)
    return pool[:limit] if limit and limit > 0 else pool


def entry_key(feature: Mapping[str, Any], position: Optional[int] = None) -> str:
    name = str(feature.get("feature_name") or feature.get("name") or "")
    raw = feature.get("pool_index")
    if isinstance(raw, int) and not isinstance(raw, bool):
        idx = raw
    elif isinstance(feature.get("input_index"), int):
        idx = int(feature["input_index"])
    elif position is not None:
        idx = position
    else:
        return name
    return f"{name}#{idx}"


def feature_iv(feature: Mapping[str, Any], iv: Mapping[str, float],
               position: Optional[int] = None) -> Optional[float]:
    value = iv.get(entry_key(feature, position))
    if value is None:
        value = feature.get("true_iv", feature.get("iv"))
    return (float(value) if isinstance(value, (int, float))
            and not isinstance(value, bool) and math.isfinite(float(value)) else None)


def ranked_indices(features: List[Dict[str, Any]], iv: Optional[Mapping[str, float]] = None,
                   ) -> List[int]:
    """Stable pool-local IV-descending order; entries without IV stay at the end."""
    values = iv or {}

    def sort_key(index: int) -> float:
        value = feature_iv(features[index], values, index)
        return -value if value is not None else float("inf")

    return sorted(range(len(features)), key=sort_key)


def load_iv_jsonl(path: Optional[str | Path]) -> Dict[str, float]:
    """Read IV values keyed by ``feature_name#pool_index`` (or input position)."""
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"IV JSONL does not exist: {p}")
    out: Dict[str, float] = {}
    for pos, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("valid") is False:
            continue
        value = row.get("iv", row.get("true_iv"))
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(float(value))):
            continue
        key = str(row.get("entry_key") or entry_key(row, pos))
        out[key] = float(value)
    return out
