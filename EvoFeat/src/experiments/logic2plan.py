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
    jfs = sorted({str(e.get("join_field") or "") for e in (subgraph.edges or [])
                  if e.get("join_field")})
    return "\n".join(jfs)


_EDGE_STEP_RE = re.compile(r"\b(?:outE|inE|bothE)\s*\(")
_QUOTED_RE = re.compile(r"['\"]([^'\"]{1,200})['\"]")


def _relations_vocab_set(subgraph) -> set:
    return {str(e.get("join_field") or "") for e in (subgraph.edges or [])
            if e.get("join_field")}


def _plan_join_fields(plan: dict) -> set:
    out: set = set()
    for st in (plan.get("steps") or []):
        g = str((st or {}).get("gremlin") or "")
        for m in _QUOTED_RE.finditer(g):
            s = m.group(1)
            if ("|" in s or "&" in s) and not s.endswith("|entity") \
                    and " " not in s:
                out.add(s)
    return out


def _max_chain_hops(plan: dict) -> int:
    steps = {str((s or {}).get("id") or ""): (s or {})
             for s in (plan.get("steps") or [])}
    depth: dict = {}

    def _d(sid: str, seen) -> int:
        if sid in depth:
            return depth[sid]
        if sid in seen or sid not in steps:
            return 0
        st = steps[sid]
        has_edge = 1 if _EDGE_STEP_RE.search(str(st.get("gremlin") or "")) else 0
        best = 0
        for d in (st.get("depends_on") or []):
            best = max(best, _d(str(d), seen | {sid}))
        depth[sid] = best + has_edge
        return depth[sid]

    return max((_d(sid, frozenset()) for sid in steps), default=0)


def _semantic_alignment(plan: dict, feat: dict, subgraph) -> tuple[list, list]:
    fatal: list = []
    warns: list = []
    vocab = _relations_vocab_set(subgraph)
    if vocab:
        bad: list = []
        variant: list = []
        for j in sorted(_plan_join_fields(plan)):
            if j in vocab:
                continue
            near = next((v for v in sorted(vocab) if j in v or v in j), None)
            (variant if near else bad).append((j, near))
        if bad:
            fatal.append("join_field is absent from the subgraph: "
                         + ", ".join(j for j, _ in bad[:3])
                         + (" ..." if len(bad) > 3 else ""))
        if variant:
            fatal.append("join_field is not an exact subgraph value and will produce zero hits: "
                         + ", ".join(f"{j} (possible variant of {v})"
                                     for j, v in variant[:3])
                         + (" ..." if len(variant) > 3 else ""))
    hops = feat.get("hops")
    chain = _max_chain_hops(plan)
    if isinstance(hops, int) and hops > 0 and chain and chain != hops:
        warns.append(f"Declared hops={hops}, but the longest dependency chain has {chain} edge-expansion steps; "
                     f"verify that the plan implements all {hops} hops")
    return fatal, warns


def validate_self_plan(feat: dict, subgraph, executor, llm) -> Candidate:
    from system.agents.consistency import diagnose_consistency
    from system.agents.synthesizer import Synthesizer
    from system.execution.eg_plan import (execute_plan, normalize_plan,
                                          render_execution_trace, sanitize_plan,
                                          validate_plan, SLOW_QUERY_ERR_PREFIX)

    name = str(feat.get("name") or "")
    logic = str(feat.get("logic") or "")
    center_vid = getattr(subgraph, "center_vid", "") or ""
    g_prefix = getattr(executor, "g_prefix", "g")
    trace: list = []
    exec_attempts = 0
    exec_error = ""
    feature_value = None
    trace_exec: list = []
    consistency_detail = None
    consistency_note = ""
    death_stage = ""
    plan = feat.get("plan")
    if not isinstance(plan, dict) or not plan.get("steps"):
        exec_error = "plan is missing or steps is empty; a baseline feature without a plan is rejected"
        death_stage = "struct"
        trace.append(TraceStep(action="Validate(struct)", obs=exec_error,
                               thought="output schema is missing the plan field"))
    else:
        plan = normalize_plan(plan)
        ok, verr = validate_plan(plan)
        if not ok:
            exec_error = f"invalid plan structure: {verr}"
            death_stage = "struct"
            trace.append(TraceStep(action="Validate(struct)", obs=verr,
                                   thought="structural validation failed"))
        else:
            fixes = sanitize_plan(plan, center_vid, g_prefix)
            if fixes:
                logger.info(f"[logic2plan] {name}: {len(fixes)} deterministic corrections")
            trace.append(TraceStep(action="Validate(struct)",
                                   obs="structural validation passed", thought="validate_plan+sanitize"))
            semantic_fatal, semantic_warns = _semantic_alignment(plan, feat, subgraph)
            if semantic_fatal:
                exec_error = "deterministic semantic alignment failed: " + "; ".join(semantic_fatal)
                death_stage = "semantic"
                logger.warning(f"[logic2plan] {name} {exec_error}"
                               f" (deterministically blocked; no repair round → counted as rejected)")
                trace.append(TraceStep(action="Validate(semantic)",
                                       obs=exec_error[:300],
                                       thought="declaration contradicts the plan; relation is absent from this subgraph"))

            value, trace_exec, err = ((None, [], exec_error) if exec_error
                                      else execute_plan(executor, plan, center_vid))
            exec_attempts = 1 if not exec_error else 0
            for _ri in range(_SLOW_EXEC_RETRIES):
                if not err or not err.startswith(SLOW_QUERY_ERR_PREFIX):
                    break
                logger.info(f"[logic2plan] {name}: slow query during execution (597 timeout); "
                            f"retrying after {_SLOW_EXEC_BACKOFF_S:.1f}s "
                            f"{_ri + 1}/{_SLOW_EXEC_RETRIES}")
                time.sleep(_SLOW_EXEC_BACKOFF_S)
                value, trace_exec, err = execute_plan(executor, plan, center_vid)
                exec_attempts += 1
                if not err:
                    trace.append(TraceStep(
                        action="Exec(retry)",
                        obs=f"slow-query retry succeeded after {exec_attempts - 1} retries",
                        thought="transient server-budget fluctuation recovered on retry"))
            if err:
                exec_error = err
                if death_stage != "semantic":
                    death_stage = "exec"
                    trace.append(TraceStep(action="Exec(fail)", obs=err[:300],
                                           thought="execution failed; baseline has no repair round and rejects the feature"))
            else:
                feature_value = value
                trace.append(TraceStep(action="Exec(ok)", obs=f"feature_value={value!r}",
                                       thought="execution succeeded; starting semantic validation"))
            _ftype_vt = _FINAL_VALUE_TYPE.get(
                str(((plan.get("final") or {}).get("type") or "")).strip().lower(), "")
            vt_errs = ([] if exec_error
                       else Synthesizer._value_type_errors(value, _ftype_vt))
            if vt_errs:
                exec_error = "; ".join(vt_errs)
                death_stage = "semantic"
                trace.append(TraceStep(action="Validate(semantic)", obs=exec_error[:300],
                                       thought="deterministic value-type check failed; baseline has no repair round and rejects the feature"))
            if not exec_error:
                summary = diagnose_consistency(
                    llm, feature=name, logic=logic,
                    evidence_chain_block=(
                        "(baseline-generated plan; no explorer evidence chain)"
                        + ("\n[DETERMINISTIC SEMANTIC ALIGNMENT WARNINGS: verify these facts strictly]\n"
                           + "\n".join(f"· {w}" for w in semantic_warns)
                           if semantic_warns else "")),
                    relations_vocab=_relations_vocab(subgraph),
                    plan=plan,
                    execution_trace_text=render_execution_trace(trace_exec),
                    feature_value=value, seed_vid=center_vid)
                if summary:
                    consistency_detail = summary
                    trace.append(TraceStep(
                        action="LLM(consistency)",
                        obs=json.dumps(summary, ensure_ascii=False, default=str),
                        thought="verbatim consistency-diagnosis LLM response"))
                    verdict = str(summary.get("consistency") or "").strip()
                    if verdict != "consistent":
                        exec_error = (f"consistency={verdict} | "
                                      f"{summary.get('inconsistency_category', '')}: "
                                      f"{summary.get('inconsistency_reason', '')}")
                        death_stage = "consistency"
                        logger.warning(
                            f"[logic2plan] {name} consistency={verdict} | "
                            f"{summary.get('inconsistency_category', '')}: "
                            f"{str(summary.get('inconsistency_reason', ''))[:160]}"
                            f" (baseline has no repair round → counted as rejected)")
                        trace.append(TraceStep(action="Validate(inconsistent)",
                                               obs=exec_error[:300],
                                               thought="consistency failed; baseline has no repair round and rejects the feature"))
                    else:
                        consistency_note = f"executed value={value!r}; consistency=consistent"
                        logger.info(
                            f"[logic2plan] {name} consistency="
                            f"{verdict} (value={value!r})")
                else:
                    exec_error = "consistency diagnosis returned no verdict"
                    death_stage = "consistency"
                    trace.append(TraceStep(action="Validate(undiagnosed)", obs=exec_error,
                                           thought="an undiagnosed feature is treated as a failure"))
    accepted = not exec_error
    final = (plan or {}).get("final") or {} if isinstance(plan, dict) else {}
    value_type = _FINAL_VALUE_TYPE.get(str(final.get("type") or ""), "count")
    cand = Candidate(
        feature_name=name,
        feature_name_cn="",
        dsl_program=json.dumps(plan if isinstance(plan, dict) else {},
                               ensure_ascii=False, indent=1),
        value_type=value_type,
        rationale=str(feat.get("rationale") or ""),
        binning="monotonic",
        trace=trace,
        synth_ok=accepted,
        validation_report=exec_error[:200],
        evidence=[],
        center_vid=center_vid,
        fraud_pattern=str(feat.get("pattern") or ""),
        evidence_chain=[],
        expected_value_type="",
        logic=logic,
        consistency_report=consistency_note
        or (f"validation failed: {exec_error[:160]}" if exec_error else "semantically consistent"),
        business_explanation=str(feat.get("rationale") or ""),
    )
    cand.gremlin_plan = plan if isinstance(plan, dict) else {}
    cand.feature_value = feature_value
    cand.execution_trace = trace_exec
    cand.exec_error = exec_error
    cand.exec_attempts = exec_attempts
    cand.consistency_detail = consistency_detail
    cand.predicted_iv = _as_iv(feat.get("predicted_iv"))
    cand._death_stage = death_stage
    cand._baseline = True
    cand._self_plan = True
    cand._skills_used = []
    cand._err_history = [exec_error] if exec_error else []
    cand._failed_plan = plan if (exec_error and isinstance(plan, dict)) else None
    cand._synth_trace = [t.to_dict() for t in trace]
    return cand
