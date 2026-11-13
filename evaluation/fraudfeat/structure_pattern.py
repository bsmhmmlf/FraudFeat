#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalize declarative graph patterns for structural diversity analysis.

This module does not execute features. It converts SQL-feature ``graph_pattern``
metadata into the internal representation consumed by the existing structural
metrics, so Gremlin and SQL implementations use exactly the same GED and statistic
signature definitions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


__all__ = ["plan_from_graph_pattern"]
_DIRECTIONS = {
    "out": ("outE", "inV"),
    "in": ("inE", "outV"),
    "both": ("bothE", "otherV"),
}
_AGGREGATES = {"sum": "sum", "mean": "mean", "avg": "mean",
               "min": "min", "max": "max", "count": "count"}


def _quoted(value: Any) -> str:
    text = str(value or "")
    if not text:
        raise ValueError("Graph-pattern relation and attribute names must not be empty")
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def plan_from_graph_pattern(pattern: Any) -> Optional[Dict[str, Any]]:
    """Return the internal structural plan, or None when no pattern is declared."""
    if pattern is None:
        return None
    if not isinstance(pattern, dict):
        raise ValueError("graph_pattern must be an object")
    raw_steps = pattern.get("steps")
    final = pattern.get("final")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("graph_pattern.steps must be a non-empty list")
    if not isinstance(final, dict) or not final.get("type"):
        raise ValueError("graph_pattern.final must declare a statistic type")
    ids = [str(step.get("id") or "") for step in raw_steps if isinstance(step, dict)]
    if len(ids) != len(raw_steps) or any(not sid for sid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Every graph-pattern step must have a unique id")
    steps: List[Dict[str, Any]] = []
    for raw in raw_steps:
        sid = str(raw["id"])
        deps = raw.get("depends_on") or []
        if isinstance(deps, str):
            deps = [deps]
        if not isinstance(deps, list) or len(deps) > 1:
            raise ValueError(f"Step {sid} must declare zero or one structural dependency")
        if deps and str(deps[0]) not in ids:
            raise ValueError(f"Step {sid} references an unknown dependency")
        anchor = "{{STEP:" + str(deps[0]) + "}}" if deps else "{{SEED}}"
        direction = str(raw.get("direction") or "out").lower()
        if direction not in _DIRECTIONS:
            raise ValueError(f"Step {sid} has unsupported direction: {direction}")
        relations = raw.get("relations")
        if relations is None:
            relations = [raw.get("relation")]
        if isinstance(relations, str):
            relations = [relations]
        if not isinstance(relations, list) or not relations or any(not r for r in relations):
            raise ValueError(f"Step {sid} must declare relation or relations")
        edge_op, vertex_op = _DIRECTIONS[direction]
        text = (f"g.V().hasId({anchor}).{edge_op}("
                + ",".join(_quoted(relation) for relation in relations) + ")")
        measure = raw.get("measure") or {}
        if isinstance(measure, str):
            measure = {"object": "vertex", "attribute": measure}
        if not isinstance(measure, dict):
            raise ValueError(f"Step {sid}.measure must be an object or attribute name")
        obj = str(measure.get("object") or "vertex").lower()
        attr = measure.get("attribute")
        if obj not in ("vertex", "edge"):
            raise ValueError(f"Step {sid}.measure.object must be vertex or edge")
        dedup = bool(raw.get("dedup"))
        if obj == "vertex":
            text += f".{vertex_op}()"
        if dedup:
            text += ".dedup()"
        if attr:
            text += f".values({_quoted(attr)})"
        agg = raw.get("agg")
        if agg:
            op = _AGGREGATES.get(str(agg).lower())
            if not op:
                raise ValueError(f"Step {sid} has unsupported aggregate: {agg}")
            text += f".{op}()"
        output = str(raw.get("output") or "id_list")
        if output not in ("id_list", "scalar", "scalar_list"):
            raise ValueError(f"Step {sid} has unsupported output: {output}")
        steps.append({"id": sid, "output": output, "gremlin": text})
    return {"steps": steps, "final": dict(final)}
