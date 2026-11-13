from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ..llm.client import LLMClient
from ..utils import get_logger


logger = get_logger("consistency")
EXPLORER_FAULT_CATEGORIES = {"insufficient_subgraph"}
SYNTHESIZER_FAULT_CATEGORIES = {
    "query_generation", "operator_approximation", "aggregation_or_metric",
    "hop_or_direction",
}
_SUMMARY_SYSTEM = """You are a senior Gremlin query reviewer and graph risk-control feature expert.
You receive the actual successful execution trace of a multi-step EasyGraph plan. Placeholders {{SEED}} and {{STEP:sX}} have been replaced by real values, and ==> marks the actual intermediate result size or scalar at each step.
Return one JSON object only, with no Markdown or extra text.
Required keys:
- detailed_feature_name: a reusable feature definition that exactly matches the executed start point, every rel, direction, hop, filter, aggregation, and measured object. Never include observed result values.
- detailed_logic: a step-by-step English description of the executed computation.
- consistency: one of consistent, partial, inconsistent, comparing actual execution with the original logic.
- inconsistency_category: one of insufficient_subgraph, query_generation, operator_approximation, aggregation_or_metric, hop_or_direction, other, none.
- inconsistency_reason: a detailed element-by-element explanation and correction direction when not consistent.
- element_comparison: an array of strings covering start subject, every relation and direction, hop count, filters, aggregation, and measured object.
Judge semantics only, never the observed feature value. A correct implementation with zero data matches is consistent. Do not recommend reversing a correct direction because the result is zero.
For ratio features, separate numerator and denominator count steps followed by Python final.ratio are the correct implementation, not extra or erroneous steps.
The schema property names are total_amount and trade_count. Amount filters and aggregates use integer fen without conversion. A schema-correct property replacing a typo in the original logic is not an inconsistency.
For relation suffixes, payer/payee describe the src role. From src use outE().inV(); from dst use inE().outV(). The mechanically injected direction block at the start of logic is the sole authority. Outgoing means the current node is src; incoming means it is dst. Conflicting residual direction wording in the original logic is subordinate to this injected block and is not an execution inconsistency.
Use insufficient_subgraph only when the implementation is semantically identical to logic but the trace proves that the stated chain is absent; cite the missing hop or edge."""


def diagnose_consistency(llm: LLMClient, *, feature: str, logic: str,
                         evidence_chain_block: str, relations_vocab: str,
                         plan: Dict[str, Any], execution_trace_text: str,
                         feature_value: Any, seed_vid: str,
                         expected_value_type: str = "") -> Optional[Dict[str, Any]]:
    parts = [
        f"[ORIGINAL FEATURE NAME — REFERENCE ONLY]\n{feature}",
        "[ORIGINAL LOGIC — CONSISTENCY BASELINE]\n" + (str(logic or "")[:2000] or "(empty)"),
        "[EVIDENCE CHAIN CONFIRMED ON THE SUBGRAPH]\n" + (evidence_chain_block or "(none)"),
        f"[EXPECTED VALUE TYPE]\n{expected_value_type or '(not declared)'}",
    ]
    if relations_vocab:
        parts.append("[ACTUAL RELATION VOCABULARY]\n" + relations_vocab)
    parts += [
        f"[SEED ACCOUNT VID]\n{seed_vid}",
        "[GREMLIN PLAN TEMPLATE]\n" + json.dumps(plan, ensure_ascii=False)[:3000],
        "[ACTUAL SUCCESSFUL EXECUTION TRACE — FACTUAL BASIS]\n"
        + (execution_trace_text[:6000] if execution_trace_text else "(none)"),
        f"[FINAL FEATURE VALUE]\n{feature_value!r}",
        "Return the required JSON object only.",
    ]
    try:
        obj = llm.complete_json(_SUMMARY_SYSTEM, "\n\n".join(parts))
    except Exception as e:
        logger.info(f"  [consistency] skipped: {type(e).__name__}: {e}")
        return None
    if isinstance(obj, dict) and (str(obj.get("detailed_feature_name", "")).strip()
                                  or str(obj.get("consistency", "")).strip()):
        return obj
    logger.info("  [consistency] empty or invalid response")
    return None


def consistency_feedback_text(summary: Dict[str, Any]) -> str:
    feedback = (
        "\n[CONSISTENCY DIAGNOSIS — CORRECT EVERY DEVIATION IN THE NEXT PLAN]"
        f"\nVerdict={summary.get('consistency', '')}; category={summary.get('inconsistency_category', '')};"
        f" reason={summary.get('inconsistency_reason', '')}"
    )
    diff = format_element_diff(summary)
    if diff:
        feedback += "\n[ELEMENT COMPARISON]\n" + diff
    detailed_logic = str(summary.get("detailed_logic") or "").strip()
    if detailed_logic:
        feedback += "\n[ACTUAL EXECUTED SEMANTICS]\n" + detailed_logic
    return feedback


def format_element_diff(summary: Dict[str, Any]) -> str:
    if not isinstance(summary, dict):
        return ""
    diff = summary.get("element_comparison")
    if isinstance(diff, list):
        return "\n".join(f"  - {item}" for item in diff if str(item).strip())
    return str(diff or "").strip()


def fault_owner(summary: Optional[Dict[str, Any]]) -> str:
    if not isinstance(summary, dict):
        return ""
    verdict = str(summary.get("consistency") or "").strip()
    if verdict in ("", "consistent"):
        return ""
    category = str(summary.get("inconsistency_category") or "").strip()
    if category in EXPLORER_FAULT_CATEGORIES:
        return "explorer"
    if category in SYNTHESIZER_FAULT_CATEGORIES:
        return "synthesizer"
    return "synthesizer"
