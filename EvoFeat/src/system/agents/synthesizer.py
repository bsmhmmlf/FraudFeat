from __future__ import annotations

import difflib
import json
import os
import re
from typing import Any

from ..llm.client import LLMClient
from ..utils import get_logger
from ..core.schema import Candidate, TraceStep, MAX_FEATURE_HOPS
from ..core.graph import node_entity, relation_info
from ..execution.eg_plan import (
    normalize_plan, sanitize_plan, validate_plan, execute_plan,
    render_execution_trace,
    SERVER_OUTAGE_ERR_PREFIX, SLOW_QUERY_ERR_PREFIX,
)
from .consistency import (
    diagnose_consistency, consistency_feedback_text, fault_owner,
)


logger = get_logger("synthesizer")
_GEN_TEMPERATURE = 0.7


_STEP_TIMEOUT_S = 30.0


_PROPOSE_SYSTEM = """You are a graph risk-control feature engineer. Given a randomly sampled target-centered subgraph without navigation filtering, propose computable features.
Use only entities and exact rel identifiers present in the subgraph. Build an ordered evidence_chain with from, rel, to, and direction out|in, using at most 5 hops and 5 evidence segments. Write self-contained logic including filters and aggregation. expected_value_type is count, flag, amount, or ratio; define numerator and denominator explicitly for ratios.
Return JSON only: {"hypotheses":[{"feature_name","feature_name_cn","fraud_pattern","logic","rationale","expected_value_type","hops","evidence_chain"}]}, with at most %s hypotheses."""

_SYSTEM_TMPL = """You are a senior graph feature engineer fluent in Apache TinkerPop Gremlin and EasyGraph. Translate the feature hypothesis into an executable multi-step Gremlin plan. Return one JSON object only.

[EASYGRAPH HUB CONTRACT]
- Every traversal starts with __G__.V(). Vertex label is entity and vertex tokens are <business_vid>|entity. Intermediate .id() returns bare business IDs; the system restores tokens.
- Edge label is relation and the business relation identifier is stored in the relation property.
- Copy each evidence-chain rel verbatim. For a vertex-producing hop, use outE().has('relation',rel).inV() from its stored src endpoint, inE().has('relation',rel).outV() from its stored dst endpoint, and bothE().has('relation',rel).otherV() only when direction is intentionally ignored.
- For a terminal amount hop, the evidence edge itself is the aggregation object: use its outE()/inE() step and directly apply values('total_amount') and the aggregate in that same step. Do not traverse inV()/outV() and then add another edge expansion.
- payer_side/payee_side names describe recorded endpoint roles. Follow each evidence-chain direction annotation exactly and never reverse it intuitively.
- Vertex properties are strings, so quote all vertex filter constants, including numeric states and timestamps. Edge total_amount and trade_count are integers.
- Use only sum(), mean(), count(), min(), and max(). There is no avg(), asDouble(), or asLong(). Division is performed only by Python final.ratio.
- All total_amount values, filters, and aggregates use integer fen without conversion.

Return this schema:
{
  "feature": "the exact input snake_case feature name",
  "feature_understanding": "one business sentence",
  "steps": [{"id":"s1","desc":"business role","depends_on":[],"output":"id_list","gremlin":"__G__.V().hasId({{SEED}}).outE().has('relation','merchant_payment_receipt_payer_side').inV().dedup().id().toList()"}],
  "final": {"type":"count","step":"s1"},
  "business_rationale": "why fraud and normal accounts differ",
  "business_explanation": "2-4 plain-English sentences derived from fraud_pattern and rationale and consistent with this plan, covering behavior, risk value, and a strategy suggestion; no jargon, node IDs, absolute amounts, or IV",
  "skills_used": ["actually applied synthesis skill IDs"]
}

Placeholders:
- {{SEED}} anchors the first step and is already a string literal; never quote it.
- {{STEP:sX}} references an upstream id_list through __G__.V().hasId(within({{STEP:s1}})); list sX in depends_on. Multiple STEP placeholders may be combined in one within/hasId.
- Step IDs start with s and contain only letters, digits, and underscores. References must exactly match defined IDs. Never invent placeholders or embed concrete VIDs.

Operators and outputs:
- Count or existence: .dedup().id().toList(), output=id_list, final type=count or exists.
- Sum, mean, min, max: values('total_amount').sum()/mean()/min()/max(), output=scalar, final type=scalar.
- Secondary reduction: collect numeric values with .toList(), output=scalar_list, final={"type":"reduce","step":"sX","op":"sum|mean|min|max"}; avg is accepted only as a Python reduce alias.
- Ratio: separate numerator and denominator id_list steps and final={"type":"ratio","numerator_step":"...","numerator_mode":"count","denominator_step":"...","denominator_mode":"count"}.
  Both ratio steps must return id_list. Supported forms include same-hop proportions and cross-layer expansion, sibling anomaly, common-neighbor overlap, and bidirectional convergence ratios.
  For cross-layer ratios, numerator_step is deeper hop k>=2 and denominator_step is shallower. Numerators may be filtered; denominators must be the complete unfiltered id_list count at a hop on the evidence chain.
  A nonempty denominator with zero numerator is a valid zero; an empty denominator means the structure is absent. Use one ratio only and never divide ratios.

Mandatory rules:
1. Anchor every chain with hasId({{SEED}}) or hasId(within({{STEP:sX}})); never scan all vertices.
2. Use one hop per step and deduplicate intermediate id_list results. The final execution path must contain exactly the evidence-chain number of main-path relation expansions, never an extra aggregation traversal. Nested where()/not() qualification traversals are filters, not additional main-path hops.
3. Filter total_amount and trade_count with integer constants; quote every vertex-property constant.
4. Cover every evidence-chain hop and make the deepest stated vertex or edge set participate in the final aggregation.
5. Match every hop, relation, direction, filter, and measured object in logic. For a terminal amount hop, reach its evidence edge and aggregate total_amount in that same step; do not reach the endpoint and then add another relation expansion.
6. Never invent filters or thresholds absent from logic or supplied facts.
7. Use anonymous __.V() inside union()/where()/local(); never use __G__.V() in a child traversal.
8. or()/and()/not() require anonymous traversal arguments.
9. Do not expose identity numbers, phone numbers, card numbers, or other PII.
10. Apply applicable synthesis playbooks on the first attempt and list their IDs in skills_used; repair attempts must follow the same playbooks.
Return JSON only."""


_PLAN_CONTRACT_HEAD = "[EASYGRAPH HUB CONTRACT]"
PLAN_CONTRACT = _PLAN_CONTRACT_HEAD + _SYSTEM_TMPL.split(_PLAN_CONTRACT_HEAD, 1)[-1]


def plan_contract_text(g_prefix: str = "g") -> str:
    return PLAN_CONTRACT.replace("__G__", g_prefix or "g")


_USER_TMPL = """Fraud-centered subgraph: center={center}, entity={center_entity}, seed VID={center_vid}.
[FEATURE HYPOTHESIS]
feature_name={name}
feature_name_cn={feature_name_cn}
fraud_pattern={fraud_pattern}
rationale={intuition}
expected_value_type={expected_value_type}
Self-contained logic that the plan must implement:
{logic}
Evidence chain with hops={hops}; cover every hop and follow its directions:
{evidence_chain_block}

[AVAILABLE ENTITIES]
{entities}

[AVAILABLE RELATIONS]
Use only these exact rel identifiers and their annotated stored directions; variants and payer_side/payee_side swaps are rejected:
{relations}

[SEED NEIGHBORHOOD FACTS]
Observed relation, direction, amount, and count samples:
{probe_block}

{skill_block}

Return the plan JSON only. feature must match exactly, steps must cover the evidence chain, and final must be compatible with expected_value_type."""


_FINAL_COMPAT_VT = {
    "count": {"count"},
    "exists": {"flag"},
    "ratio": {"ratio"},
    "scalar": {"amount", "count"},
    "reduce": {"amount"},
}
_FINAL_VALUE_TYPE = {
    "count": "count", "exists": "flag", "ratio": "ratio",
    "scalar": "amount", "reduce": "amount",
}
_JF_HAS_RE = re.compile(r"has\(\s*['\"](?:join_field|relation)['\"]\s*,\s*['\"]([^'\"]+)['\"]")
_JF_WITHIN_RE = re.compile(r"has\(\s*['\"](?:join_field|relation)['\"]\s*,\s*within\(([^)]*)\)")
_EXPANSION_RE = re.compile(r"\.(outE|inE|bothE)\(")


class Synthesizer:
    def __init__(self, llm: LLMClient | None = None, max_repair: int = 3,
                 semantic_check: bool = True, executor=None):
        self.llm = llm or LLMClient()
        self.max_repair = max_repair
        self.semantic_check = semantic_check
        self.executor = executor
        if executor is not None:
            self.g_prefix = executor.g_prefix
        else:
            space = os.environ.get("SYSTEM_EG_SPACE", "")
            if not space:
                try:
                    from ..utils import load_config
                    space = str(load_config("execution.yaml").get("space", ""))
                except Exception:
                    space = ""
            self.g_prefix = f"g_{space}" if space else "g"
        self._system = _SYSTEM_TMPL.replace("__G__", self.g_prefix)

    @property
    def exec_enabled(self) -> bool:
        return self.executor is not None

    def propose_features(self, subgraph, max_features: int = 5) -> list[dict]:
        entities, relations = self._subgraph_vocab(subgraph)
        edge_lines = []
        for e in (getattr(subgraph, "edges", None) or [])[:200]:
            s = e.get("src") or e.get("s")
            d = e.get("dst") or e.get("t")
            raw_rel = e.get("join_field") or e.get("rel") or ""
            rel = (relation_info(raw_rel) or {}).get("en") or raw_rel
            if s and d:
                edge_lines.append(f"{s} -[{rel}]-> {d}")
        nodes_line = "; ".join(
            f"{sid}({(nd or {}).get('entity', '?')})"
            for sid, nd in list(subgraph.nodes.items())[:120])
        user = (f"Target center:{subgraph.center}\n"
                f"Nodes:{nodes_line}\n"
                f"Edges (src -[rel]-> dst is storage direction):\n"
                + "\n".join(edge_lines) + "\n"
                + f"Entity vocabulary:{entities}\nRelation vocabulary:{relations}\n")
        try:
            resp = self.llm.complete_json(_PROPOSE_SYSTEM % max_features, user,
                                          temperature=0.5)
        except Exception as e:
            logger.warning(f"[Ablation 1: direct features] LLM call failed ({e}); no output for target")
            return []
        hyps = resp.get("hypotheses") if isinstance(resp, dict) else None
        if not isinstance(hyps, list):
            logger.warning("[Ablation 1: direct features] LLM returned no hypotheses list; no output for target")
            return []
        out = [h for h in hyps
               if isinstance(h, dict) and (h.get("feature_name") or h.get("name"))]
        logger.info(f"[Ablation 1: direct features] Generated {len(out)} hypotheses from random subgraph"
                    f" (limit {max_features})")
        return out[:max_features]

    def synthesize(self, hypothesis: dict, subgraph, pool: list[Candidate],
                   trace: list[TraceStep],
                   index: int = 0, total: int = 1,
                   skills: list | None = None,
                   skill_memory=None,
                   preset_plan: dict | None = None) -> Candidate:
        _synth_start = len(trace)
        center_entity = node_entity(subgraph.nodes.get(subgraph.center, {}))
        feat_name = (hypothesis.get("feature_name") or hypothesis.get("name") or "unnamed")
        logger.info(f"[Synthesize] Hypothesis {index+1}/{total}: {feat_name}")
        center_vid = subgraph.center_vid or subgraph.center
        chain = hypothesis.get("evidence_chain") or []
        try:
            declared_hops = int(hypothesis.get("hops") or 0)
        except (TypeError, ValueError):
            declared_hops = 0
        try:
            evidence_hops = max([len(chain)] + [int(s.get("hop") or i + 1)
                                                for i, s in enumerate(chain)], default=0)
        except (TypeError, ValueError):
            evidence_hops = len(chain)
        hops = evidence_hops or declared_hops
        evt = str(hypothesis.get("expected_value_type") or "count").lower()
        chain_block = self._chain_block(subgraph, chain, evt)
        entities, relations = self._subgraph_vocab(subgraph)
        probe_block = self._probe_block(subgraph)
        skill_block = self._skill_block(skills, skill_memory)
        user = _USER_TMPL.format(
            center=subgraph.center, center_entity=center_entity,
            center_vid=center_vid,
            name=feat_name,
            feature_name_cn=hypothesis.get("feature_name_cn", "") or "",
            fraud_pattern=hypothesis.get("fraud_pattern", hypothesis.get("risk_pattern", "")) or "",
            intuition=(hypothesis.get("rationale") or hypothesis.get("intuition")
                       or hypothesis.get("discrimination") or ""),
            expected_value_type=hypothesis.get("expected_value_type", ""),
            logic=hypothesis.get("logic", "") or "(not provided; use evidence_chain)",
            hops=hops,
            evidence_chain_block=chain_block,
            entities=entities,
            relations=relations,
            probe_block=probe_block,
            skill_block=skill_block,
        )
        max_rounds = self.max_repair + 1
        best_plan: dict | None = None
        sem_err: list[str] = ["not synthesized yet"]
        err_history: list[str] = []
        last_plan: dict | None = None
        raw = None
        last_declared: list[str] = []
        accepted = False
        llm_passed = False
        feature_value: Any = None
        execution_trace: list = []
        exec_error = ""
        exec_attempts = 0
        consistency_detail: dict = {}
        consistency_note = ""
        explorer_fault = False
        for _r in range(max_rounds):
            if _r == 0:
                prompt = user
            else:
                prompt = self._build_repair_prompt(
                    sem_err, feat_name, hops,
                    err_history=err_history, last_plan=last_plan,
                    chain_block=chain_block, entities=entities,
                    relations=relations, probe_block=probe_block,
                    expected_value_type=hypothesis.get("expected_value_type", "") or "",
                    logic=hypothesis.get("logic", "") or "",
                    fraud_pattern=hypothesis.get("fraud_pattern", hypothesis.get("risk_pattern", "")) or "",
                    rationale=(hypothesis.get("rationale") or hypothesis.get("intuition") or ""),
                    last_explanation=(str(raw.get("business_explanation") or "").strip()
                                      if isinstance(raw, dict) else ""))
                _kept = [str(e)[:120] for e in sem_err[:2]]
                _line = (f"{'initial' if _r == 1 else f'repair {_r - 1}'}: " + "; ".join(_kept))
                _dropped = len(sem_err) - len(_kept)
                if _dropped > 0:
                    _line += f" ({_dropped} additional errors omitted; see current feedback)"
                err_history.append(_line)
                if skills:
                    prompt += "\n\n" + skill_block
                logger.info(f"[Synthesize] Hypothesis {index+1}/{total}, repair {_r}/{self.max_repair}"
                            f" (feedback: {'; '.join(str(e)[:60] for e in sem_err[:3])})")
                trace.append(TraceStep(
                    action="ReAct(repair)",
                    obs="; ".join(str(e)[:200] for e in sem_err[:4]),
                    thought=f"self-repair round {_r}: iterate from validation feedback"))
            if _r == 0 and preset_plan is not None:
                raw = dict(preset_plan)
                trace.append(TraceStep(
                    action="PlanFromExplorer",
                    obs=f"{len(raw.get('steps') or [])} steps",
                    thought="W/O-SYN: the explorer supplied the plan; this module only validates it"))
                logger.info(f"[Synthesize] Hypothesis {index+1}/{total}: explorer-supplied plan"
                            f" (ablation 2, no generation call); validating")
            else:
                raw = self.llm.complete_json(self._system, prompt,
                                             temperature=_GEN_TEMPERATURE)
            if raw is None:
                logger.warning(f"[Synthesize] Hypothesis {index+1}/{total}: no LLM response; using deterministic fallback plan")
                sem_err = []
                break
            trace.append(TraceStep(
                action="LLM(synthesize)",
                obs=json.dumps(raw, ensure_ascii=False, default=str),
                thought=f"verbatim synthesis LLM response for round {_r + 1}/{max_rounds}"))
            if isinstance(raw, dict):
                u = raw.get("skills_used") or []
                last_declared = ([str(x).strip() for x in u if str(x).strip()]
                                 if isinstance(u, list) else [])
                if last_declared:
                    trace.append(TraceStep(
                        action="SynthSkill(declare)",
                        obs=f"declared skills {last_declared}",
                        thought="the model judged these skills applicable to this feature"))
            if not isinstance(raw, dict) or not raw.get("steps"):
                sem_err = ["No plan produced: steps is missing or empty"]
                continue
            plan = normalize_plan(raw)
            fixes = sanitize_plan(plan, center_vid, self.g_prefix)
            if fixes:
                logger.info(f"      Auto-corrections: {len(fixes)}: " + "; ".join(fixes[:3]))
            if self.semantic_check:
                rel_fixes = self._autocorrect_relation_names(plan, chain, subgraph)
                if rel_fixes:
                    logger.info(f"      Relation-name corrections: {len(rel_fixes)}: "
                                + "; ".join(rel_fixes[:3]))
                    trace.append(TraceStep(
                        action="AutoFix(rel)",
                        obs="; ".join(rel_fixes[:4]),
                        thought="deterministic correction of invented relation names"))
            last_plan = plan
            dir_warns = self._autocorrect_direction(plan, subgraph,
                                                    hypothesis.get("evidence_chain") or [])
            if dir_warns:
                logger.info(f"      Direction warnings: {len(dir_warns)}: " + "; ".join(dir_warns[:3]))
            ok, verr = validate_plan(plan)
            trace.append(TraceStep(
                action="Validate(struct)",
                obs="syntactic validation passed" if ok else f"syntactic validation failed: {verr}",
                thought="checking the normalized plan before trial execution"))
            if not ok:
                sem_err = dir_warns + [f"Syntactic validation failed: {verr}"]
                continue
            if not self.exec_enabled:
                det_err = (self.check_semantic_alignment(
                    plan, chain, hops,
                    hypothesis.get("expected_value_type", ""), subgraph)
                    if self.semantic_check else [])
                if det_err:
                    sem_err = dir_warns + det_err
                    trace.append(TraceStep(
                        action="Validate(semantic)", obs="; ".join(det_err[:4]),
                        thought="deterministic semantic validation failed without a graph executor"))
                    continue
                best_plan = plan
                accepted = True
                llm_passed = True
                consistency_note = "syntactic validation passed (no graph execution)"
                break
            exec_attempts += 1
            value, trace_exec, exec_err = execute_plan(
                self.executor, plan, center_vid,
                step_timeout_s=_STEP_TIMEOUT_S)
            execution_trace = trace_exec
            if exec_err:
                if exec_err.startswith(SERVER_OUTAGE_ERR_PREFIX):
                    logger.warning(f"      [Execute] {exec_err[:160]}")
                exec_error = exec_err
                if exec_err.startswith(SLOW_QUERY_ERR_PREFIX):
                    sem_err = [f"{exec_err}",
                               "The query exceeded the server budget. Reduce upstream sets with stricter filters and earlier deduplication, avoid deep traversals nested inside where(), and split multiple filters into separate steps. Execution details:\n"
                               + render_execution_trace(trace_exec)[:1500]]
                    trace.append(TraceStep(
                        action="Exec(slow)",
                        obs=exec_err[:300],
                        thought="query exceeded the server budget; return convergence guidance to repair"))
                else:
                    sem_err = [f"Execution failed; correct the plan: {exec_err}",
                               "Execution details:\n" + render_execution_trace(trace_exec)[:1500]]
                    trace.append(TraceStep(
                        action="Exec(fail)",
                        obs=exec_err[:300],
                        thought="live execution failed; return the error and execution details to repair"))
                continue
            feature_value = value
            trace.append(TraceStep(
                action="Exec(ok)",
                obs=f"feature_value={value!r}",
                thought="live execution succeeded; starting semantic validation"))
            det_err = []
            if self.semantic_check:
                det_err = self.check_semantic_alignment(
                    plan, chain, hops,
                    hypothesis.get("expected_value_type", ""), subgraph)
                det_err.extend(self._value_type_errors(
                    value, hypothesis.get("expected_value_type", "")))
            trace.append(TraceStep(
                action="Validate(semantic)",
                obs=("; ".join(str(e)[:200] for e in det_err[:4])
                     if det_err else "deterministic semantic validation passed"),
                thought="checking evidence relations and scalar value type after successful trial execution"))
            if det_err:
                sem_err = dir_warns + det_err
                continue
            summary = diagnose_consistency(
                self.llm, feature=feat_name,
                logic=hypothesis.get("logic", "") or "",
                evidence_chain_block=chain_block,
                relations_vocab=relations,
                plan=plan,
                execution_trace_text=render_execution_trace(trace_exec),
                feature_value=value, seed_vid=center_vid,
                expected_value_type=hypothesis.get("expected_value_type", ""))
            if summary:
                consistency_detail = summary
                trace.append(TraceStep(
                    action="LLM(consistency)",
                    obs=json.dumps(summary, ensure_ascii=False, default=str),
                    thought="verbatim consistency-diagnosis LLM response"))
                verdict = str(summary.get("consistency") or "").strip()
                logger.info(f"      [Consistency diagnosis] verdict={verdict or 'N/A'} | "
                            f"detailed_feature_name: {summary.get('detailed_feature_name', '')}")
                if verdict != "consistent":
                    owner = fault_owner(summary)
                    explorer_fault = owner == "explorer"
                    exec_error = (f"consistency={verdict} | "
                                  f"{summary.get('inconsistency_category', '')}: "
                                  f"{summary.get('inconsistency_reason', '')}")
                    sem_err = [consistency_feedback_text(summary)]
                    trace.append(TraceStep(
                        action="Validate(inconsistent)",
                        obs=exec_error[:300],
                        thought=f"consistency diagnosis failed (owner={'explorer' if explorer_fault else 'synthesizer'}); "
                                f"returning details to repair"))
                    continue
                consistency_note = f"executed value={value!r}; consistency=consistent"
            else:
                exec_error = "consistency diagnosis unavailable"
                sem_err = ["Semantic diagnosis returned no verdict; regenerate a plan whose steps "
                           "explicitly follow the logic so that it can be validated"]
                trace.append(TraceStep(
                    action="Validate(undiagnosed)",
                    obs=exec_error,
                    thought="LLM-based semantic diagnosis returned no verdict; treated as a failed check"))
                continue
            best_plan = plan
            exec_error = ""
            sem_err = []
            accepted = True
            llm_passed = True
            break
        fallback = not accepted
        if fallback:
            logger.warning(f"[Synthesize] Hypothesis {index+1}/{total}: synthesis and repair failed; "
                           f"marking as a negative sample (no fallback plan)")
            best_plan = last_plan or {}
            feature_value, execution_trace = None, []
            trace.append(TraceStep(
                action="Validate(failed)",
                obs="repair budget exhausted; synthesis failed and becomes evidence for skill evolution and failure diagnosis",
                thought="consistency gate failed; process-level negative sample"))
        if skill_memory is not None and last_declared:
            for sid in last_declared:
                upd = getattr(skill_memory, "update_stats", None)
                if callable(upd):
                    upd(sid, hit=llm_passed)
        final = (best_plan or {}).get("final") or {}
        value_type = _FINAL_VALUE_TYPE.get(str(final.get("type") or ""), "")
        if not value_type:
            value_type = (hypothesis.get("expected_value_type", "") or "").strip().lower()
            if value_type not in ("count", "flag", "amount", "ratio"):
                value_type = "count"
        if fallback:
            consistency_report = "synthesis failed (repair budget exhausted; negative sample)"
        else:
            consistency_report = consistency_note or "semantically consistent"
        logger.info(f"Synthesized feature {feat_name} -> Gremlin plan ({len((best_plan or {}).get('steps') or [])} steps)"
                    f": {consistency_report}")
        rationale = (raw.get("business_rationale") or raw.get("rationale")
                     or hypothesis.get("intuition", "")) if isinstance(raw, dict) \
            else hypothesis.get("intuition", "")
        cand = Candidate(
            feature_name=feat_name,
            feature_name_cn=hypothesis.get("feature_name_cn", "") or "",
            dsl_program=json.dumps(best_plan, ensure_ascii=False, indent=1),
            value_type=value_type,
            rationale=rationale or hypothesis.get("intuition", ""),
            binning="monotonic",
            trace=list(trace),
            synth_ok=not fallback,
            validation_report="; ".join(str(e)[:200] for e in sem_err),
            evidence=[{"a": s.get("from"), "rel": s.get("rel"),
                       "b": s.get("to")} for s in chain] or hypothesis.get("evidence", []) or [],
            center_vid=center_vid,
            fraud_pattern=hypothesis.get("fraud_pattern", hypothesis.get("risk_pattern", "")) or "",
            evidence_chain=chain,
            expected_value_type=hypothesis.get("expected_value_type", ""),
            logic=hypothesis.get("logic", "") or "",
            consistency_report=consistency_report,
            business_explanation=(
                str(raw.get("business_explanation") or "").strip()
                if isinstance(raw, dict) else "") or
            (str(raw.get("feature_understanding") or "").strip()
             if isinstance(raw, dict) else ""),
        )
        cand.gremlin_plan = best_plan or {}
        cand.feature_value = feature_value
        cand.execution_trace = execution_trace
        cand.exec_error = exec_error
        cand.exec_attempts = exec_attempts
        cand.consistency_detail = consistency_detail
        self._apply_summary_names(cand, consistency_detail)
        cand._skills_used = list(last_declared)
        cand._synth_fallback = fallback
        cand._explorer_fault = explorer_fault
        cand._synth_trace = [t.to_dict() for t in cand.trace[_synth_start:]]
        if fallback or err_history:
            cand._synth_vocab = {"entities": entities, "relations": relations}
            cand._err_history = list(err_history)
            cand._failed_plan = last_plan if fallback else None
        trace.append(TraceStep(
            action="Synthesize+Validate",
            obs=f"feature={cand.feature_name}, value_type={cand.value_type}, "
                f"exec={bool(exec_attempts)}",
            thought=f"generated Gremlin plan ({consistency_report})"))
        return cand

    @staticmethod
    def _apply_summary_names(cand: Candidate, summary: dict) -> None:
        if not isinstance(summary, dict) or not summary:
            return
        detail_logic = str(summary.get("detailed_logic") or "").strip()
        if detail_logic:
            cand.consistency_detail = dict(cand.consistency_detail or {})
            cand.consistency_detail["detailed_logic"] = detail_logic

    @staticmethod
    def _skill_block(skills: list | None, skill_memory) -> str:
        if not skills:
            return "[REUSABLE SYNTHESIS SKILLS] none"
        if skill_memory is not None:
            pt = getattr(skill_memory, "playbook_text", None)
            if callable(pt):
                try:
                    txt = pt()
                except Exception:
                    txt = None
                if txt:
                    return txt
        lines = ["[SYNTHESIS PLAYBOOKS] Apply relevant entries and declare their IDs in skills_used"]
        for s in skills:
            lines.append(f"### {s.anchor or s.id} [{s.id}]")
            lines.append(f"- when: {s.when}")
            lines += [f"- {p}" for p in s.procedure]
        return "\n".join(lines)

    @staticmethod
    def _collect_real_rels(chain: list, subgraph) -> set[str]:
        real_rels: set[str] = set()
        if subgraph is not None:
            for e in getattr(subgraph, "edges", []) or []:
                jf = e.get("join_field") or e.get("rel")
                if jf:
                    real_rels.add(str(jf))
            for es in getattr(subgraph, "adj", {}).values():
                for e in es:
                    for k in ("rel", "rel_name"):
                        if e.get(k):
                            real_rels.add(str(e[k]))
        for seg in (chain or []):
            if isinstance(seg, dict):
                for k in ("join_field", "rel"):
                    if seg.get(k):
                        real_rels.add(str(seg[k]))
        return real_rels

    @staticmethod
    def _split_jf(jf: str) -> tuple[str, str]:
        base, sep, suffix = jf.rpartition("|")
        return (base if sep else jf), (suffix if sep else "")

    @staticmethod
    def _jf_match(miss: str, cand: str) -> tuple[float, bool]:
        mb, ms = Synthesizer._split_jf(miss)
        cb, cs = Synthesizer._split_jf(cand)
        if not cs:
            return 0.0, False
        base_sim = difflib.SequenceMatcher(None, mb.lower(), cb.lower()).ratio()
        if "&" in mb or "&" in cb:
            sa, sb = set(mb.lower().split("&")), set(cb.lower().split("&"))
            jac = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
            base_sim = max(base_sim, jac)
        mbl, cbl = mb.lower(), cb.lower()
        if mbl and cbl and (mbl in cbl or cbl in mbl):
            contain = 0.7 + 0.25 * min(len(mbl), len(cbl)) / max(len(mbl), len(cbl))
            base_sim = max(base_sim, contain)
        if ms == cs:
            return base_sim, False
        role_pair = {ms, cs} == {"payer", "payee"}
        ent_rev = ("&" in ms and "&" in cs
                   and sorted(x.strip().lower() for x in ms.split("&"))
                   == sorted(x.strip().lower() for x in cs.split("&")))
        if role_pair or ent_rev:
            return base_sim - 0.05, True
        return 0.0, False

    @staticmethod
    def _flip_direction_for_jf(gremlin: str, jf: str) -> str | None:
        expansions = list(re.finditer(r"\.(outE|inE|bothE)\(\)", gremlin))
        if len(expansions) != 1:
            return None
        m = expansions[0]
        if m.group(1) == "bothE":
            return gremlin
        idx = gremlin.find(jf, m.end())
        if idx < 0:
            return None
        seg = gremlin[idx + len(jf):]
        v = re.search(r"\.(inV|outV)\(\)", seg)
        if v is None:
            return None
        flip = {"outE": "inE", "inE": "outE", "inV": "outV", "outV": "inV"}
        v_abs = idx + len(jf) + v.start()
        return (gremlin[:m.start()] + f".{flip[m.group(1)]}()"
                + gremlin[m.end():v_abs]
                + f".{flip[v.group(1)]}()"
                + seg[v.end():])

    @staticmethod
    def _required_traverse_dir(hint: str) -> str | None:
        if "outE().inV()" in hint:
            return "outE"
        if "inE().outV()" in hint:
            return "inE"
        return None

    @staticmethod
    def _step_traverse_dir(gremlin: str, jf: str) -> str | None:
        expansions = list(re.finditer(r"\.(outE|inE|bothE)\(\)", gremlin))
        if len(expansions) != 1:
            return None
        m = expansions[0]
        if m.group(1) == "bothE":
            return None
        if gremlin.find(jf, m.end()) < 0:
            return None
        return m.group(1)

    @staticmethod
    def _autocorrect_direction(plan: dict, subgraph, chain: list) -> list[str]:
        warnings: list[str] = []
        if not chain:
            return warnings
        steps = plan.get("steps") or []
        last_step_idx = -1
        for seg in chain:
            jf = seg.get("join_field") or seg.get("rel")
            if not jf:
                continue
            match_idx = None
            for i in range(last_step_idx + 1, len(steps)):
                g = str(steps[i].get("gremlin") or "")
                if jf in g:
                    match_idx = i
                    break
            if match_idx is None:
                continue
            last_step_idx = match_idx
            g = str(steps[match_idx].get("gremlin") or "").strip()
            d = seg.get("direction")
            if d == "out":
                required = "outE"
            elif d == "in":
                required = "inE"
            else:
                hint = Synthesizer._seg_direction_hint(subgraph, seg.get("from"),
                                                       seg.get("to"), jf)
                required = Synthesizer._required_traverse_dir(hint)
            cur = Synthesizer._step_traverse_dir(g, jf)
            if required and cur and required != cur:
                warnings.append(
                    f"Direction warning: segment '{jf}' currently uses {cur}(), but the stored evidence-chain "
                    f"direction requires {required}(). If the payer/payee role semantics differ intentionally, decide "
                    f"explicitly; if this is a direction slip, flip the segment ({cur} <-> {required})")
        return warnings

    @staticmethod
    def _autocorrect_relation_names(plan: dict, chain: list, subgraph) -> list[str]:
        real = Synthesizer._collect_real_rels(chain, subgraph)
        if not real:
            return []
        corrections: list[str] = []
        for st in (plan.get("steps") or []):
            g = str(st.get("gremlin") or "")
            if not g:
                continue
            used: list[str] = list(dict.fromkeys(_JF_HAS_RE.findall(g)))
            for body in _JF_WITHIN_RE.findall(g):
                for x in re.findall(r"['\"]([^'\"]+)['\"]", body):
                    if x not in used:
                        used.append(x)
            for jf in used:
                hit = any(jf in rr or rr in jf for rr in real if rr)
                if hit:
                    continue
                best, best_score, best_flip = None, 0.0, False
                for rr in real:
                    if not rr:
                        continue
                    sc, flip = Synthesizer._jf_match(jf, rr)
                    if flip and sc < 0.7:
                        continue
                    if sc > best_score:
                        best, best_score, best_flip = rr, sc, flip
                if best is None or best_score < 0.6:
                    continue
                new_g = g.replace(jf, best)
                if best_flip:
                    flipped = Synthesizer._flip_direction_for_jf(new_g, best)
                    if flipped is None:
                        continue
                    new_g = flipped
                corrections.append(
                    f"'{jf}' -> '{best}'" + (" (suffix roles swapped; direction flipped accordingly)" if best_flip else ""))
                g = new_g
            st["gremlin"] = g
        return corrections

    @staticmethod
    def _value_type_errors(value: Any, expected_value_type: str = "") -> list[str]:
        if value is None or (isinstance(value, str) and not value.strip()):
            return []
        if isinstance(value, (list, tuple, set, dict)):
            return [f"Semantic validation: the query output is a {type(value).__name__}, "
                    f"not a scalar; reduce the final step to a single value"]
        try:
            num = float(value)
        except (TypeError, ValueError):
            return [f"Semantic validation: the query output {str(value)[:60]!r} is not numeric; "
                    f"the feature must produce a numeric scalar"]
        evt = (expected_value_type or "").strip().lower()
        if evt == "flag" and num not in (0.0, 1.0):
            return [f"Semantic validation: expected_value_type=flag but the output is {num}; "
                    f"a flag must be 0 or 1"]
        if evt == "count" and (num < 0 or num != int(num)):
            return [f"Semantic validation: expected_value_type=count but the output is {num}; "
                    f"a count must be a non-negative integer"]
        if evt in ("amount", "ratio") and num < 0:
            return [f"Semantic validation: expected_value_type={evt} but the output is negative ({num})"]
        return []

    @staticmethod
    def _top_level_expansions(gremlin: str) -> tuple[int, bool]:
        """Count main-chain edge expansions, excluding nested qualification traversals."""
        count = 0
        parens = 0
        quote = ""
        escaped = False
        for i, ch in enumerate(gremlin):
            if quote:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    quote = ""
                continue
            if ch in ("'", '"'):
                quote = ch
            elif ch == "(":
                parens += 1
            elif ch == ")":
                parens = max(0, parens - 1)
            elif ch == "." and parens == 0 and _EXPANSION_RE.match(gremlin, i):
                count += 1
        opaque = bool(re.search(r"\b(?:local|union)\s*\(", gremlin)
                      and _EXPANSION_RE.search(gremlin))
        return count, opaque

    @staticmethod
    def _main_path_depth(plan: dict[str, Any]) -> int | None:
        """Return the deepest final-reachable path; None means an opaque branch traversal."""
        by_id = {str(st.get("id")): st for st in (plan.get("steps") or []) if st.get("id")}
        memo: dict[str, int | None] = {}
        visiting: set[str] = set()

        def depth(step_id: str) -> int | None:
            if step_id in memo:
                return memo[step_id]
            if step_id in visiting:
                return None
            st = by_id.get(step_id)
            if st is None:
                return 0
            visiting.add(step_id)
            local_hops, opaque = Synthesizer._top_level_expansions(str(st.get("gremlin") or ""))
            deps = st.get("depends_on") or []
            if isinstance(deps, str):
                deps = [deps]
            parent_depths = [depth(str(dep)) for dep in deps]
            visiting.discard(step_id)
            if opaque or any(d is None for d in parent_depths):
                memo[step_id] = None
            else:
                memo[step_id] = max(d for d in parent_depths if d is not None) + local_hops \
                    if parent_depths else local_hops
            return memo[step_id]

        final = plan.get("final") or {}
        if final.get("type") == "ratio":
            terminals = [final.get("numerator_step"), final.get("denominator_step")]
        elif final.get("step"):
            terminals = [final.get("step")]
        else:
            terminals = list(by_id)
        depths = [depth(str(step_id)) for step_id in terminals if step_id]
        if not depths or any(d is None for d in depths):
            return None
        return max(d for d in depths if d is not None)

    @staticmethod
    def check_semantic_alignment(plan: dict, chain: list,
                                 hops: int, expected_value_type: str = "",
                                 subgraph=None) -> list[str]:
        errs: list[str] = []
        steps = plan.get("steps") or []
        chain_hops = int(hops or 0)
        evt = (expected_value_type or "").strip().lower()
        actual_depth = Synthesizer._main_path_depth(plan)
        if chain_hops > MAX_FEATURE_HOPS:
            errs.append(f"Semantic alignment: evidence_chain has {chain_hops} hops, exceeding the "
                        f"global maximum of {MAX_FEATURE_HOPS}")
        if actual_depth is not None and actual_depth > MAX_FEATURE_HOPS:
            errs.append(f"Semantic alignment: the final execution path has {actual_depth} main-path hops, "
                        f"exceeding the global maximum of {MAX_FEATURE_HOPS}")
        if chain_hops >= 2:
            if actual_depth is None:
                expansions = sum(1 for st in steps
                                 if _EXPANSION_RE.search(str(st.get("gremlin") or "")))
                if expansions < chain_hops:
                    errs.append(f"Semantic alignment: evidence_chain has {chain_hops} hops, but the plan has only "
                                f"{expansions} edge-expansion steps; all hops and the deepest signal must participate")
            elif actual_depth != chain_hops:
                amount_hint = (" Aggregate terminal amount on the final evidence edge in that same step."
                               if evt == "amount" and actual_depth > chain_hops else "")
                errs.append(f"Semantic alignment: the final execution path has {actual_depth} main-path hops, "
                            f"but evidence_chain has exactly {chain_hops}; remove extra traversals or restore "
                            f"missing hops.{amount_hint}")
        real_rels = Synthesizer._collect_real_rels(chain, subgraph)
        used_jfs: list[str] = []
        for st in steps:
            g = str(st.get("gremlin") or "")
            used_jfs.extend(_JF_HAS_RE.findall(g))
            for within_body in _JF_WITHIN_RE.findall(g):
                used_jfs.extend(re.findall(r"['\"]([^'\"]+)['\"]", within_body))
        for jf in dict.fromkeys(used_jfs):
            hit = any(jf in rr or rr in jf for rr in real_rels if rr)
            if not hit:
                errs.append(f"Semantic alignment: relation '{jf}' is absent from the evidence chain and "
                            f"observed subgraph relations; it appears to be invented")
        ftype = str((plan.get("final") or {}).get("type") or "").strip()
        evt = (expected_value_type or "").strip().lower()
        if ftype and evt:
            compat = _FINAL_COMPAT_VT.get(ftype)
            if compat is None:
                errs.append(f"Semantic alignment: invalid final.type='{ftype}'; "
                            f"expected scalar/count/exists/reduce/ratio")
            elif evt in ("count", "flag", "amount", "ratio") and evt not in compat:
                errs.append(f"Semantic alignment: final.type='{ftype}' is incompatible with expected_value_type="
                            f"'{evt}'; choose a reduction that produces {evt}")
        return errs

    @staticmethod
    def _seg_direction_hint(subgraph, a: str, b: str, jf: str) -> str:
        if subgraph is not None and jf and jf != "?":
            for e in getattr(subgraph, "edges", []) or []:
                if (e.get("join_field") or e.get("rel")) != jf:
                    continue
                if e.get("src") == a and e.get("dst") == b:
                    return f"direction:{a}=src endpoint (forward) -> use outE().inV()"
                if e.get("src") == b and e.get("dst") == a:
                    return f"direction:{a}=dst endpoint (reverse) -> use inE().outV()"
        info = relation_info(jf)
        fe, te = info.get("from_entity") or "", info.get("to_entity") or ""
        fs, ts = info.get("from_short") or "", info.get("to_short") or ""
        if fe and te:
            a_ent = node_entity(subgraph.nodes.get(a) or {}) if subgraph is not None else ""
            b_ent = node_entity(subgraph.nodes.get(b) or {}) if subgraph is not None else ""
            if a_ent == fe and b_ent == te:
                return f"direction:{a} ({fs or fe})=src endpoint (forward) -> use outE().inV()"
            if a_ent == te and b_ent == fe:
                return f"direction:{a} ({ts or te})=dst endpoint (reverse) -> use inE().outV()"
            return (f"direction:registry convention {fs or fe}=src, {ts or te}=dst"
                    f" (from src use outE().inV(); from dst use inE().outV())")
        suffix = jf.rsplit("|", 1)[1] if "|" in (jf or "") else ""
        if "&" in suffix:
            src_ent, dst_ent = suffix.split("&", 1)
            return (f"direction:suffix convention {src_ent.strip()}=src, {dst_ent.strip()}=dst"
                    f" (from src use outE().inV(); from dst use inE().outV())")
        return ""

    @staticmethod
    def _chain_block(subgraph, chain: list, expected_value_type: str = "") -> str:
        lines = []
        is_amount = str(expected_value_type or "").strip().lower() == "amount"
        for i, s in enumerate(chain):
            a, b = s.get("from"), s.get("to")
            edge = subgraph.edge_between(a, b) if subgraph is not None else None
            rel = s.get("rel") or (edge or {}).get("rel_id") or s.get("join_field") or "?"
            rn = (edge or {}).get("rel_name") or ""
            rn_txt = f" ({rn})" if rn and rn != rel else ""
            note = s.get("note") or ""
            d = s.get("direction")
            amount_tail = is_amount and i == len(chain) - 1
            if amount_tail and d in ("out", "in"):
                edge_op = "outE()" if d == "out" else "inE()"
                endpoint_op = "inV()" if d == "out" else "outV()"
                direction = (f"direction:{a}={d} endpoint; terminal amount hop -> use "
                             f"{edge_op}.has('relation',rel) and aggregate total_amount in this same step; "
                             f"do not call {endpoint_op} or add another relation")
            elif d in ("out", "in"):
                direction = (f"direction:{a}=src endpoint (forward) -> use outE().has('relation',rel).inV()" if d == "out"
                             else f"direction:{a}=dst endpoint (reverse) -> use inE().has('relation',rel).outV()")
            elif amount_tail:
                direction = ("terminal amount hop -> follow the stored direction with outE()/inE() and aggregate "
                             "total_amount in this same step; do not traverse to the endpoint or add another relation")
            else:
                direction = Synthesizer._seg_direction_hint(subgraph, a, b, rel)
            dir_txt = f" | {direction}" if direction else ""
            lines.append(f"  {i+1}. {a} -[{rel}]{rn_txt}-> {b}"
                         f" (hop={s.get('hop')}{dir_txt})"
                         f"{(': ' + note) if note else ''}")
        return "\n".join(lines) or "  (none)"

    @staticmethod
    def _subgraph_vocab(subgraph) -> tuple[str, str]:
        ents = sorted({node_entity(nd) for nd in subgraph.nodes.values()} - {"unknown"})
        rel_pairs: dict[str, str] = {}
        for es in subgraph.adj.values():
            for e in es:
                jf = e.get("rel_id") or e.get("rel")
                rn = e.get("rel_name")
                if jf and jf not in rel_pairs:
                    rel_pairs[jf] = rn or ""
        entities = "\n".join(f"  {e}" for e in ents) or "  (none)"
        name_count: dict[str, int] = {}
        for rn in rel_pairs.values():
            if rn:
                name_count[rn] = name_count.get(rn, 0) + 1
        rel_lines = []
        for jf, rn in sorted(rel_pairs.items()):
            suffix = jf.rsplit("|", 1)[1] if "|" in jf else ""
            role = {"payer": "direction:payer(src)->payee(dst)",
                    "payee": "direction:payee(src)->payer(dst)"}.get(suffix)
            info = relation_info(jf)
            fs, ts = info.get("from_short") or "", info.get("to_short") or ""
            if role is None and fs:
                role = f"direction:{fs}(src)->{ts}(dst)" if ts else f"direction:src={fs}"
            if role is None and "&" in suffix:
                src_ent, dst_ent = suffix.split("&", 1)
                role = f"direction:{src_ent.strip()}(src)->{dst_ent.strip()}(dst)"
            head = f"  {jf} => {rn}" if (rn and rn != jf) else f"  {jf}"
            if rn and rn != jf and name_count.get(rn, 0) > 1 and (fs or ts):
                head += f"[{fs or '?'}->{ts or '?'}]"
            rel_lines.append(head + (f" ({role})" if role else ""))
        relations = "\n".join(rel_lines) or "  (none)"
        return entities, relations

    @staticmethod
    def _probe_block(subgraph, sample: int = 8) -> str:
        center = subgraph.center
        if not center:
            return "  (none)"
        lines: list[str] = []
        src_set = {(e.get("src"), e.get("dst"), e.get("join_field"))
                   for e in getattr(subgraph, "edges", []) or []}
        for e in (subgraph.adj.get(center, []) or [])[:sample]:
            raw_rel = e.get("rel") or "?"
            rel = e.get("rel_id") or raw_rel
            direction = "out (seed=src)" if (center, e.get("nbr"), raw_rel) in src_set \
                else "in (seed=dst)"
            amt, cnt = e.get("total_amount"), e.get("trade_count")
            nums = []
            if amt is not None:
                nums.append(f"amount={amt}")
            if cnt is not None:
                nums.append(f"count={cnt}")
            lines.append(f"  {rel} | {direction} | other={e.get('nbr')}"
                         + (" | " + ", ".join(nums) if nums else ""))
        return "\n".join(lines) or "  (no seed edges)"

    def _build_repair_prompt(self, sem_err: list[str], feat_name: str = "",
                             hops: int = 0, *, err_history: list[str] | None = None,
                             last_plan: dict | None = None,
                             chain_block: str = "", entities: str = "",
                             relations: str = "", probe_block: str = "",
                             expected_value_type: str = "", logic: str = "",
                             fraud_pattern: str = "", rationale: str = "",
                             last_explanation: str = "") -> str:
        parts: list[str] = []
        if chain_block:
            parts.append(f"[EVIDENCE CHAIN: cover all {hops} hops with exact relations]\n{chain_block}")
        if entities or relations:
            parts.append(f"[AVAILABLE ENTITIES]\n{entities}\n[AVAILABLE RELATIONS: exact rel values only]\n{relations}")
        if probe_block:
            parts.append(f"[SEED NEIGHBORHOOD FACTS]\n{probe_block}")
        spec_lines: list[str] = []
        if expected_value_type:
            spec_lines.append(f"expected_value_type={expected_value_type}; final reduction must match")
        if logic:
            spec_lines.append(f"Original logic and consistency baseline: {logic[:800]}")
        if fraud_pattern:
            spec_lines.append(f"fraud_pattern={fraud_pattern}")
        if rationale:
            spec_lines.append(f"rationale={rationale}")
        if spec_lines:
            parts.append("[FEATURE SPECIFICATION]\n" + "\n".join(spec_lines))
        if last_plan:
            plan_txt = json.dumps(last_plan, ensure_ascii=False, indent=1)
            parts.append(f"[FAILED PLAN: make targeted corrections]\n{plan_txt[:2500]}")
        if last_explanation:
            parts.append(f"[PREVIOUS BUSINESS EXPLANATION]\n{last_explanation}")
        if err_history:
            parts.append("[PREVIOUS FAILURES: change approach rather than oscillating]\n"
                         + "\n".join(err_history[-4:]))
        parts.append("[CURRENT VALIDATION FEEDBACK]\n"
                     + "\n".join(str(e)[:1200] for e in sem_err[:4]))
        parts.append(f"Return the corrected plan JSON only, regenerating business_explanation so that it describes the corrected query. "
                     f"feature must remain {feat_name or '?'}; "
                     f"cover exactly {hops} main-path hops with exact relations and authoritative directions, "
                     f"and match expected_value_type. For a terminal amount hop, aggregate its evidence edge "
                     f"in that same step without traversing to its endpoint or adding another relation.")
        return "\n\n".join(parts)
