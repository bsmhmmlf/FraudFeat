from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from system.core.schema import Candidate, TraceStep
from system.utils import get_logger

logger = get_logger("experiments.logic2plan")


_FINAL_VALUE_TYPE = {
    "count": "count", "exists": "flag", "ratio": "ratio",
    "scalar": "amount", "reduce": "amount",
}


def _as_iv(v) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return max(0.0, x)


_SLOW_EXEC_RETRIES = 2
_SLOW_EXEC_BACKOFF_S = 1.5


def extract_features(result: dict, baseline: str) -> list[dict]:
    try:
        if baseline == "manual":
            analysis = result.get("analysis")
        elif baseline == "stat":
            fa = (result.get("analysis_result") or {}).get("final_analysis")
            analysis = json.loads("\n".join(fa) if isinstance(fa, list)
                                  else (fa if isinstance(fa, str) else ""))
        elif baseline == "comm":
            ana = result.get("analysis")
            analysis = json.loads("\n".join(ana) if isinstance(ana, list)
                                  else (ana if isinstance(ana, str) else ""))
        else:
            raise ValueError(f"Unknown baseline {baseline}")
        if isinstance(analysis, str):
            analysis = json.loads(analysis)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"[logic2plan] {baseline} result parsing failed ({e}); no features for this target")
        return []

    def _pick(f: dict, pattern: str) -> dict | None:
        if not isinstance(f, dict) or not f.get("name"):
            return None
        return {
            "name": str(f["name"]),
            "logic": str(f.get("logic") or ""),
            "rationale": str(f.get("rationale") or ""),
            "hops": f.get("hops"),
            "evidence_chain": f.get("evidence_chain"),
            "expected_value_type": f.get("expected_value_type"),
            "value_range": str(f.get("value_range") or ""),
            "plan": f.get("plan"),
            "pattern": pattern,
            "predicted_iv": _as_iv(f.get("predicted_iv")),
        }

    if not isinstance(analysis, dict):
        return []
    if isinstance(analysis.get("features"), list):
        return [x for x in (_pick(f, "") for f in analysis["features"]) if x]
    out: list[dict] = []
    for pat in (analysis.get("fraud_patterns") or analysis.get("risk_patterns") or []):
        if not isinstance(pat, dict):
            continue
        pattern = str(pat.get("description") or pat.get("name") or "")
        out += [x for x in (_pick(f, pattern) for f in (pat.get("features") or []))
                if x]
    return out


def _relations_vocab(subgraph) -> str:
    relations = sorted({str(e.get("relation") or "") for e in (subgraph.edges or [])
                        if e.get("relation")})
    return "\n".join(relations)


_EDGE_STEP_RE = re.compile(r"\b(?:outE|inE|bothE)\s*\(")
_QUOTED_RE = re.compile(r"['\"]([^'\"]{1,200})['\"]")


def _relations_vocab_set(subgraph) -> set:
    return {str(e.get("relation") or "") for e in (subgraph.edges or [])
            if e.get("relation")}


def _plan_relations(plan: dict) -> set:
    from system.agents.synthesizer import _RELATION_HAS_RE, _RELATION_WITHIN_RE
    out = set()
    for step in plan.get("steps") or []:
        query = str(step.get("gremlin") or "")
        out.update(_RELATION_HAS_RE.findall(query))
        for body in _RELATION_WITHIN_RE.findall(query):
            out.update(re.findall(r"['\"]([^'\"]+)['\"]", body))
    return out


def _max_chain_hops(plan: dict) -> int:
    from system.agents.synthesizer import Synthesizer
    return Synthesizer._main_path_depth(plan) or 0


def _semantic_alignment(plan: dict, feat: dict, subgraph) -> tuple[list, list]:
    from system.agents.synthesizer import Synthesizer
    errors = Synthesizer.check_semantic_alignment(
        plan, feat.get("evidence_chain") or [], int(feat.get("hops") or 0),
        str(feat.get("expected_value_type") or ""), subgraph)
    return errors, []


def validate_self_plan(feat: dict, subgraph, executor, llm) -> Candidate:
    """Validate a prompt-baseline plan through the Synthesizer's common path.

    A preset plan and max_repair=0 mean no synthesis/generation call and no
    repair opportunity. The shared normalizer, trial execution, deterministic
    semantic checks and LLM diagnosis are identical to the agent path.
    """
    from system.agents.synthesizer import Synthesizer

    name = str(feat.get("name") or "")
    logic = str(feat.get("logic") or "")
    rationale = str(feat.get("rationale") or "")
    plan = feat.get("plan")
    chain = feat.get("evidence_chain")
    expected = str(feat.get("expected_value_type") or "").strip().lower()
    hops = feat.get("hops")
    issues = []
    if not isinstance(plan, dict) or not plan.get("steps"):
        issues.append("plan is missing or steps is empty")
    if expected not in ("count", "flag", "amount", "ratio"):
        issues.append("expected_value_type must be declared independently of the plan")
    if not isinstance(hops, int) or isinstance(hops, bool) or hops < 1:
        issues.append("hops must be a positive integer")
    if (not isinstance(chain, list) or not chain
            or not all(isinstance(seg, dict) and seg.get("from") and seg.get("to")
                       and seg.get("rel") and seg.get("direction") in ("out", "in", "both")
                       and isinstance(seg.get("hop"), int)
                       and not isinstance(seg.get("hop"), bool) and seg["hop"] >= 1
                       for seg in chain)):
        issues.append("evidence_chain must contain independently declared typed traversal segments")
    elif isinstance(hops, int) and not isinstance(hops, bool):
        if max(len(chain), max(seg["hop"] for seg in chain)) != hops:
            issues.append("hops disagrees with the independently declared evidence_chain")
    if not name or not logic:
        issues.append("feature name and independent computation logic are required")
    if executor is None:
        issues.append("trial execution requires a graph executor")

    if issues:
        error = "; ".join(issues)
        cand = Candidate(
            feature_name=name, feature_name_cn="",
            dsl_program=json.dumps(plan if isinstance(plan, dict) else {},
                                   ensure_ascii=False, indent=1),
            value_type=expected if expected in ("count", "flag", "amount", "ratio") else "count",
            rationale=rationale, binning="monotonic", synth_ok=False,
            trace=[TraceStep(action="Validate(struct)", obs=error,
                             thought="missing or inconsistent independent validation inputs")],
            validation_report=error, evidence=[],
            center_vid=getattr(subgraph, "center_vid", "") or "",
            fraud_pattern=str(feat.get("pattern") or ""),
            evidence_chain=chain if isinstance(chain, list) else [],
            expected_value_type=expected, logic=logic,
            consistency_report="validation failed: " + error,
            business_explanation=rationale,
        )
        cand.gremlin_plan = plan if isinstance(plan, dict) else {}
        cand.feature_value, cand.execution_trace = None, []
        cand.exec_error, cand.exec_attempts = error, 0
        cand.consistency_detail = {}
        cand._death_stage = "struct"
        cand._err_history = [error]
        cand._failed_plan = plan if isinstance(plan, dict) else None
        cand._synth_trace = [step.to_dict() for step in cand.trace]
    else:
        hypothesis = {
            "feature_name": name,
            "feature_name_cn": "",
            "logic": logic,
            "hops": hops,
            "evidence_chain": chain,
            "expected_value_type": expected,
            "rationale": rationale,
            "intuition": rationale,
            "fraud_pattern": str(feat.get("pattern") or ""),
        }
        # The verifier may normalize its local copy; never change cached inputs.
        import copy
        preset = copy.deepcopy(plan)
        preset.setdefault("business_rationale", rationale)
        preset.setdefault("business_explanation", rationale)
        verifier = Synthesizer(llm=llm, max_repair=0,
                               semantic_check=True, executor=executor)
        cand = verifier.synthesize(hypothesis, subgraph, [], [], preset_plan=preset)
        cand._death_stage = ""
        if not cand.synth_ok:
            cand._death_stage = "semantic"
            for step in reversed(cand.trace):
                action = str(getattr(step, "action", ""))
                obs = str(getattr(step, "obs", ""))
                if action in ("Validate(inconsistent)", "Validate(undiagnosed)"):
                    cand._death_stage = "consistency"
                    break
                if action in ("Exec(fail)", "Exec(slow)"):
                    cand._death_stage = "exec"
                    break
                if action == "Validate(semantic)" and obs != "deterministic semantic validation passed":
                    cand._death_stage = "semantic"
                    break
                if action == "Validate(struct)" and "failed" in obs:
                    cand._death_stage = "struct"
                    break

    cand.predicted_iv = _as_iv(feat.get("predicted_iv"))
    cand._baseline = True
    cand._self_plan = True
    cand._skills_used = []
    return cand
