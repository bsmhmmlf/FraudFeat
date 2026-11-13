from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any

from ..llm.client import LLMClient
from ..utils import get_logger
from ..core.graph import Subgraph, node_entity, is_account, relation_info
from ..core.schema import ExplorerState, Skill, TraceStep
from ..execution.eg_plan import normalize_plan, validate_plan


logger = get_logger("explorer")
_NAV_SYSTEM = """You are a graph risk-control investigator exploring multiple hops around a fraud-centered account.
Each step shows the accumulated visited subgraph and expandable neighbors. Storage direction src->dst is a graph fact, not necessarily money flow. Transaction amounts are integer fen.
Candidates are ranked by two local statistics: bridge-source count (from: retained nodes adjacent to the candidate) and hub degree (new_deg: number of unseen neighbors). Other candidate fields: entity, hop, rel, attrs, and transaction amount/count.
Infer identifier strength from observable evidence only. Natural persons, individual customers, bank cards, and phone numbers are usually stronger than generic associations; multiple sources can indicate suspicious sharing, but very large public hubs may be noise. Explain this judgment in why.
Use paradigm and navigation playbooks to complete partial structures. If no existing paradigm fits, continue exploring the genuinely new structure.
Actions:
1. Expand(ids, why): add real candidates and fetch their neighbors.
2. Discard(ids, why): remove valueless candidates or visited nodes; center cannot be discarded.
3. Revisit(ids, why): restore discarded nodes only when new evidence changes the earlier decision.
4. Focus(id): add the unvisited neighbors of one retained node directly to the retained set.
5. Stop(why): finish only when enough information is available or all remaining candidates are valueless.
Make at least N Expand/Discard decisions. Cover different sources, relations, and entity types. Prefer multi-source bridges, strong shared identifiers, useful hubs, and hop>=2 structures over leaves. Never invent IDs.
Return JSON only: {"observation":"one concrete structural observation with node IDs","reason":"coverage and selection strategy","actions":[{"action":"Expand|Discard|Revisit|Focus|Stop","ids":[],"why":""}]}."""

_HYP_SYSTEM = """You are the feature-hypothesis stage of a graph risk-control Explorer. Use the actual visited path, reviewed subgraph, observations, and detected signals to propose 2-4 interpretable features.
Match the observed network to applicable paradigms and hypothesis playbooks, but treat detected signals as hints rather than a whitelist. New structures outside existing paradigms are valuable.
Mandatory rules:
1. Define structure by an ordered evidence_chain of real relations. Use at least three nodes, at least two real edges, and at least one edge not incident to the center, except for the amount exception below.
2. Adjacent evidence segments must connect. from/to IDs must exist and hop must reflect distance from the center. Supply a path with real entity types and relations.
3. Follow the depth rule in the user message exactly. The deepest stated hop must participate in aggregation. Amount features aggregating an edge attribute may cover only the hop containing that edge. Every value type, including amount, has a global maximum of 5 hops and at most 5 evidence segments.
4. Reject trivial one-hop counts or totals without structural, scale, or rarity constraints.
5. State the fraud_pattern and its rationale: why fraud accounts should differ from normal accounts.
6. Hypotheses must be semantically distinct and use different relation sequences.
7. logic must be self-contained English. Use full entity names and exact rel identifiers from evidence_chain. Never use anonymous node IDs, invented set variables, references to other hypotheses, or payer/payee suffixes as node identities.
8. Do not write direction words in the LLM-authored portion of logic. The system injects authoritative storage directions from the subgraph. State only rel and the destination entity at each hop; the final candidate retains the injected directions.
9. Produce one scalar per target account and state the expected normal-account behavior.
Output types:
- count: deduplicated count of a constrained object set.
- flag: 1 if a constrained object exists, otherwise 0.
- amount: sum, mean, min, or max of an edge numeric attribute; state the operator.
- ratio: divide two deduplicated counts and define both sets explicitly.
[OP1] Cross-layer expansion ratio: complete or filtered count at hop k divided by the complete unfiltered count at earlier hop m, 2<=k<=max_hops and 1<=m<k.
[OP2] Sibling anomaly multiple: filtered sibling count at one hop divided by the complete unfiltered count at that hop.
[OP3] Common-neighbor overlap ratio: shared neighbors from an earlier-hop node divided by that node's complete neighbor count.
[OP4] Bidirectional convergence ratio: next-hop neighbors with additional incoming edges from same-hop nodes divided by all next-hop outbound neighbors.
Every ratio denominator must be a complete unfiltered id_list count on the evidence chain. Never use a rare filtered subset as denominator. If no valid denominator exists, do not force a ratio.
Each hypothesis contains a feature name and two groups of fields.
Group 1 (computation): logic (attribute filters, measured entities/relations/attributes, deduplication rules, aggregation operations and their composition), evidence_chain [{from,rel,to,hop,note}] grounded in observed relations, expected_value_type, path, and hops.
Group 2 (risk): fraud_pattern and rationale for its potential association with fraud.
Name fields: feature_name (snake_case) and feature_name_cn (short business-readable English display name stating metric, object, business relationship, and relevant filters without graph jargon or node IDs).
Return JSON only: {"hypotheses":[...]}"""

_NAV_USER_TMPL = """Target fraud account: {center} ({center_entity}).
Progress: hop {step}/{max_hops}; visited={visited_n}; focus={focus}.
Subgraph: {total_nodes} nodes/{total_edges} edges; coverage={coverage}%.{coverage_hint}
Visible cap={cap}; expansion quota N={quota}.
Known structural signals: {real_signal_hint}
{feedback_block}
{saturation_block}
{paradigm_block}
{skill_block}
{discarded_block}
{current_view}
Expand about {quota} nodes across different sources, relations, and entities. Discard valueless candidates and Revisit only with new evidence. Return JSON only."""

_HYP_USER_TMPL = """Multi-hop investigation for fraud account {center} ({center_entity}): {visited_n} reviewed nodes, maximum depth {max_depth}.
{depth_rule_block}
[VISITED SUBGRAPH SUMMARY]
{structure_summary}
[ACTUAL TRAVERSAL]
{traversal_view}
{nav_notes_block}
{feedback_block}{paradigm_block}{skill_block}
[DETECTED STRUCTURAL SIGNALS — HINTS ONLY]
{real_signal_block}
Propose 2-4 distinct risk features satisfying the system rules and {chain_req_inline}. Return JSON only."""

_DEPTH_BLOCK_STATIC = """Actual maximum depth is {required_hops}. Every hypothesis must have hops={required_hops}, at least that many connected real evidence edges, and final aggregation involving the deepest hop."""

_DEPTH_BLOCK_LIVE = """Actual maximum depth is {required_hops}. Each hypothesis must have {min_hops}<=hops<={required_hops}, at least hops connected evidence segments, and aggregation involving its deepest hop."""

_CHAIN_INLINE_STATIC = "evidence_chain>={required_hops}, one edge excluding the center, path, and hops={required_hops}"
_CHAIN_INLINE_LIVE = "connected evidence_chain with one edge excluding the center, path, and {min_hops}<=hops<={required_hops}"

_HYP_SYSTEM_LIVE_APPEND = """
Live mode overrides strict equality: {min_hops}<=hops<=actual maximum depth, with at least hops evidence segments."""

_NAV_SYSTEM_LIVE_APPEND = """
Live mode fetches neighbors on Expand and consumes query budget. For overflow groups, use Expand or Discard with groups such as {{"action":"Expand","groups":[{{"key":"G1","top":10}}]}}. Stop before depth {min_hops} is rejected."""


_REVIEW_SYSTEM = """Review the complete explored subgraph from a global perspective. Correct local exploration errors with Discard for visited noise, Revisit for discarded nodes with new evidence, and Expand for missing boundary nodes with structural value.
Keep changes minimal, use only real IDs, never discard center, and respect the user-provided restoration limit.
Return JSON only: {"review":"global structural judgment, valuable paths, and gaps","actions":[{"action":"Discard|Revisit|Expand","ids":[],"why":""}]}. Empty actions are valid when no correction is needed."""

_REVIEW_USER_TMPL = """Exploration for fraud account {center} ({center_entity}) ended with {visited_n} visited nodes and maximum depth {max_depth}.
[VISITED SUBGRAPH SUMMARY]
{structure_summary}
[VISITED NODE REVIEW]
{visited_view}
[UNEXPANDED BOUNDARY CANDIDATES]
{frontier_view}
{discarded_block}
Return review and actions. Discard applies to visited nodes, Revisit to discarded nodes, and Expand to boundary candidates; Revisit+Expand must total at most {add_cap}. Return JSON only."""


class Explorer:
    DEFAULT_BUDGETS = [1, 50, 250, 500, 1000, 2000]

    _PII_ATTR_SUBSTR = ("cre_id", "id_card", "cert", "card_no", "card",
                        "phone", "mobile", "password", "pwd", "token")

    def __init__(self, llm: LLMClient | None = None, max_hops: int = 3,
                 branch: int = 10, top_k: int = 12,
                 budget_schedule: list[int] | None = None,
                 auto_topup: bool = False, view_cap: int = 400,
                 node_cap: int = 60, nbr_cap: int = 8,
                 strict_depth: bool = False,
                 global_review: bool = True,
                 source=None, min_hops: int = 0,
                 emit_plan: bool = False):
        self.llm = llm or LLMClient()
        self.max_hops = int(max_hops)
        self.branch = branch
        self.top_k = top_k
        self.budget_schedule = budget_schedule
        self.auto_topup = auto_topup
        self.view_cap = view_cap
        self.node_cap = node_cap
        self.nbr_cap = nbr_cap
        self.strict_depth = strict_depth
        self.global_review = global_review
        self._cur_step = 0
        self.source = source
        self.min_hops = int(min_hops)

        self.emit_plan = bool(emit_plan)

    def hop_budgets(self) -> list[int]:
        sched = sorted({int(x) for x in (self.budget_schedule or self.DEFAULT_BUDGETS)
                        if int(x) > 0})
        if not sched:
            sched = list(self.DEFAULT_BUDGETS)
        while len(sched) < self.max_hops + 1:
            sched.append(sched[-1] * 5)
        return sched[: self.max_hops + 1]

    def explore(self, subgraph: Subgraph, skills: list[Skill] | None = None,
                state: ExplorerState | None = None,
                online_feedback: str | None = None,
                skill_memory: Any | None = None,
                pool_summary: str = "",
                diversity_block: str = "",
                paradigm_block: str = "",
                guidance_block: str = "",
                random_nodes: bool = False) -> dict:
        trace: list[TraceStep] = []
        center = subgraph.center
        center_entity = node_entity(subgraph.nodes.get(center, {}))
        if state is None:
            state = ExplorerState()
        if not state.visited:
            state.visited.add(center)
        visited = state.visited
        trace.append(TraceStep(action="SampleAnchors",
                               obs=f"center={center} (round={state.round})",
                               thought="anchor at the risky center account and investigate hop by hop"))

        if (self._live() and (self.min_hops or random_nodes)
                and state.round == 0):
            if not self.source.prepare(subgraph, state):
                logger.info(f"[Explore] {center}: shallow graph or failed probe; skipping target without a negative sample")
                trace.append(TraceStep(action="SkipTarget",
                                       obs=f"reachable_depth<{self.min_hops}",
                                       thought="shallow graph skipped without creating a negative sample"))
                return {"hypotheses": [], "trace": trace, "signals": [],
                        "visited": sorted(visited),
                        "effective_subgraph": None, "state": state}
        real_signals = subgraph.detect_signals()
        real_signal_block = self._real_signal_block(real_signals)
        real_signal_hint = self._signal_hint(real_signals)
        feedback_block = guidance_block or ""
        pool_summary = ""
        if guidance_block:
            logger.info(f"[Explore] Injecting coverage guidance g_(t-1): {len(guidance_block)} characters")
        if online_feedback:
            feedback_block += self._render_online_feedback_block(online_feedback)
        nav_skill_block = self._skill_block_for("navigate", skill_memory)
        if not paradigm_block:
            paradigm_block = self._skill_block_for("paradigm", skill_memory)
        budgets = self.hop_budgets()

        total_nodes = len(subgraph.nodes)
        total_edges = len(getattr(subgraph, "edges", []) or [])
        large_graph = total_nodes > 500
        stop_topup_ratio = 0.8 if large_graph else 0.5
        loop_topup_ratio = 0.7 if large_graph else 0.5
        logger.info(f"[Explore] Cumulative visible-node budgets per hop: {budgets}"
                    f" (max_hops={self.max_hops}, auto_topup={'on' if self.auto_topup else 'off'}, "
                    f"strict_depth={'on' if self.strict_depth else 'off'}, "
                    f"large_graph={'on' if large_graph else 'off'} ({total_nodes} nodes), "
                    f"round state={'inherited' if state.round > 0 else 'new'})")
        nav_notes: list[str] = []
        recent: list[str] = []
        if random_nodes:
            visited.update(subgraph.nodes.keys())
            trace.append(TraceStep(action="BudgetedInclusion",
                                   obs=f"{len(visited)} nodes included under the per-hop node budget",
                                   thought="random sampling when candidates exceed the remaining budget"))
        for step in range(0 if random_nodes else self.max_hops):
            self._cur_step = step
            prev_visited = set(visited)
            cap = budgets[min(step + 1, len(budgets) - 1)]
            quota = max(0, cap - len(visited))
            if quota <= 0:
                logger.info(f"[Explore] Hop {step+1}/{self.max_hops}: visible-node cap {cap} reached; stopping expansion")
                break
            view_text, has_expand, n_cand = self._navigation_view(
                subgraph, state, quota, recent)
            if not has_expand:
                logger.info(f"[Explore] Hop {step+1}/{self.max_hops}: no expandable neighbors; stopping expansion")
                trace.append(TraceStep(action="Stop",
                                       obs="leaf reached; no expandable neighbors",
                                       thought="stop expansion"))
                break
            coverage = round(len(visited) / max(1, total_nodes) * 100, 1)
            coverage_hint = ("Warning: this graph is large. Low coverage can systematically miss bridge or mediator nodes. "
                             "Diversify expansion within the quota across sources, relations, and entity types. "
                             "Do not stop early merely because the current view looks ordinary."
                             if large_graph else "")
            logger.info(f"[Explore] Hop {step+1}/{self.max_hops}: {len(visited)} nodes visited -> "
                        f"visible cap {cap}, expansion quota {quota}, {n_cand} neighbors; requesting LLM actions")
            user = _NAV_USER_TMPL.format(
                center=center, center_entity=center_entity,
                step=step + 1, max_hops=self.max_hops, current_view=view_text,
                visited_n=len(visited), cap=cap, quota=quota,
                total_nodes=total_nodes, total_edges=total_edges,
                coverage=coverage, coverage_hint=coverage_hint,
                focus=state.focus or center,
                real_signal_hint=real_signal_hint,
                feedback_block=feedback_block,
                saturation_block=pool_summary or "",
                paradigm_block=paradigm_block or "",
                skill_block=nav_skill_block or "(no matching navigation skills)",
                discarded_block=self._render_discarded_block(state))
            if self._live():
                pv = self.source.pending_view(subgraph)
                if pv:
                    user += "\n\n" + pv
            nav_system = (_NAV_SYSTEM + _NAV_SYSTEM_LIVE_APPEND.format(min_hops=self.min_hops)
                          if self._live() and self.min_hops else _NAV_SYSTEM)
            decision = self.llm.complete_json(nav_system, user, temperature=0.8)
            if not isinstance(decision, dict):
                logger.warning("Navigation decision parsing failed; stopping expansion")
                break
            obs_text = str(decision.get("observation", "") or "").strip()
            reason_text = str(decision.get("reason", "") or "").strip()
            if obs_text or reason_text:
                nav_notes.append(f"[hop {step+1}] observation: {obs_text}; reason: {reason_text}")
            trace.append(TraceStep(
                action="LLM(navigate)",
                obs=json.dumps(decision, ensure_ascii=False, default=str),
                thought=reason_text or "verbatim navigation decision for this hop"))
            actions = decision.get("actions") or []
            if decision.get("done") or (
                    isinstance(actions, list) and any(
                        isinstance(a, dict) and a.get("action") == "Stop" for a in actions)):
                topped = 0
                if self.auto_topup and len(visited) < cap * stop_topup_ratio:
                    topped, ids = self._topup(subgraph, visited, quota,
                                              exclude=state.discarded)
                    recent = ids
                    logger.warning(
                        f"[Explore] Hop {step+1}/{self.max_hops}: LLM stopped early with only "
                        f"{len(visited)-topped}/{cap} budget used -> added {topped} nodes automatically")
                logger.info(f"[Explore] Hop {step+1}/{self.max_hops}: LLM found sufficient information; "
                            f"stopping expansion ({len(visited)} nodes visited), "
                            + ("entering global review and fraud-pattern derivation" if self.global_review
                               else "entering fraud-pattern derivation"))
                trace.append(TraceStep(
                    action="Stop",
                    obs=(f"hop {step+1}: LLM found sufficient information (auto_topup={topped})"
                         + (f" | observation: {obs_text}" if obs_text else "")),
                    thought=reason_text or "stop expansion and derive fraud patterns"))
                break
            if not isinstance(actions, list) or not actions:
                logger.warning("LLM returned no valid actions; treating as Stop")
                trace.append(TraceStep(action="Stop",
                                       obs=f"hop {step+1}: no valid action returned",
                                       thought="treated as STOP"))
                break
            stats = self._apply_actions(subgraph, state, actions, quota,
                                        step=step, trace=trace,
                                        recent=recent, obs_text=obs_text,
                                        reason_text=reason_text,
                                        nav_notes=nav_notes)
            added, topped = stats["added"], stats["topped"]
            discarded = stats.get("discarded", 0)
            recent = stats["recent"]
            if self.auto_topup and (added + discarded) < n_cand * loop_topup_ratio:
                topped2, ids = self._topup(subgraph, visited, quota - added,
                                           exclude=state.discarded)
                if topped2:
                    state.add_log(step + 1, "Topup", ids, "unused quota filled by bridge/hub ranking")
                    recent = ids
                    logger.warning(f"[Explore] Hop {step+1}/{self.max_hops}: LLM used only {added}/{quota} quota"
                                   f" -> added {topped2} nodes automatically")
                    added += topped2
                    topped += topped2

            if self._live():
                newly = visited - prev_visited
                if newly:
                    self.source.ensure_expanded(subgraph, newly)
            logger.info(f"[Explore] Hop {step+1}/{self.max_hops}: expanded {added}/{quota} (including {topped} automatic additions)"
                        f", {len(visited)} nodes visited in total")
            if added == 0:
                logger.warning("LLM selected no valid neighbors and no candidates remain for automatic addition; stopping expansion")
                break
        if self.global_review and not random_nodes:
            self._global_review(subgraph, state, trace, nav_notes)
        if self._live() and (self.min_hops or random_nodes):
            if not random_nodes:
                self.source.ensure_expanded(subgraph, visited)
            depth = self._visited_depth(subgraph, visited)
            if depth < self.min_hops:
                before = depth
                deepened = (0 if random_nodes else
                            self._auto_deepen(subgraph, state, visited, trace))
                depth = self._visited_depth(subgraph, visited)
                if deepened:
                    nav_notes.append(
                        f"[DEPTH GATE] the chain reached only {before} hops; the system extended it "
                        f"to {depth} hops (+{deepened} nodes) to satisfy the minimum depth {self.min_hops}")
                if depth < self.min_hops:
                    logger.info(f"[Explore] Depth after extension is still {depth} < {self.min_hops}; "
                                f"skipping shallow graph without a negative sample")
                    trace.append(TraceStep(action="SkipTarget",
                                           obs=f"reachable depth {depth} after extension",
                                           thought="shallow graph skipped without creating a negative sample"))
                    return {"hypotheses": [], "trace": trace, "signals": real_signals,
                            "visited": sorted(visited),
                            "effective_subgraph": None, "state": state}
            self.source.finalize(subgraph, visited)
        traversal_text = self._traversal_view(subgraph, visited)
        live_mode = self._live() and bool(self.min_hops)
        if live_mode:
            max_depth = self._visited_depth(subgraph, visited)
            required_hops = max(self.min_hops, max_depth)
        else:
            max_depth = max((subgraph.nodes.get(n, {}).get("first_hop") or 0)
                            for n in visited) if visited else 0
            required_hops = max(2, max_depth)
        if self.max_hops:
            required_hops = min(required_hops, self.max_hops)
            max_depth = min(max_depth, self.max_hops)
        hop_req_txt = (f"hops(>={self.min_hops} and <={required_hops})" if live_mode
                       else f"hops(exactly {required_hops})")
        chain_req_txt = ("evidence_chain uses real edges and has at least the declared number of hops" if live_mode
                         else f"evidence_chain uses real edges and has at least {required_hops} segments")
        nav_notes_block = ("[INVESTIGATION OBSERVATIONS AND REASONING]\n"
                           + "\n".join(nav_notes) + "\n") if nav_notes else ""

        skill_block = self._skill_block_for("hypothesize", skill_memory)
        logger.info(f"[Explore] Deriving fraud patterns from {len(visited)} visited nodes, maximum depth {max_depth}: "
                    f"{hop_req_txt} + {chain_req_txt}")
        if live_mode:
            hyp_system = _HYP_SYSTEM + _HYP_SYSTEM_LIVE_APPEND.format(min_hops=self.min_hops)
            depth_rule_block = _DEPTH_BLOCK_LIVE.format(required_hops=required_hops,
                                                        min_hops=self.min_hops)
            chain_req_inline = _CHAIN_INLINE_LIVE.format(required_hops=required_hops,
                                                         min_hops=self.min_hops)
        else:
            hyp_system = _HYP_SYSTEM
            depth_rule_block = _DEPTH_BLOCK_STATIC.format(required_hops=required_hops)
            chain_req_inline = _CHAIN_INLINE_STATIC.format(required_hops=required_hops)
        if self.emit_plan:
            hyp_system = hyp_system + "\n\n" + self._plan_contract_block()
            logger.info("[Explore] Ablation 2: generate features and gremlin_plan together under the plan contract "
                        "(no later synthesis; hypotheses without a plan are rejected)")
        hyp_user = _HYP_USER_TMPL.format(
            center=center, center_entity=center_entity,
            visited_n=len(visited), max_depth=max_depth,
            required_hops=required_hops,
            depth_rule_block=depth_rule_block,
            chain_req_inline=chain_req_inline,
            structure_summary=self._structure_summary(subgraph, visited),
            traversal_view=traversal_text,
            nav_notes_block=nav_notes_block,
            feedback_block=feedback_block,
            paradigm_block=paradigm_block or "",
            skill_block=skill_block or "(no matching skills)",
            real_signal_block=real_signal_block)
        if self.emit_plan:
            hyp_user = hyp_user + "\n" + self._plan_output_block()
        hypotheses: list[dict] = []
        resp = self.llm.complete_json(hyp_system, hyp_user, temperature=0.9)
        from_llm = bool(resp and isinstance(resp, dict) and "hypotheses" in resp)
        if from_llm:
            hypotheses = resp["hypotheses"]
            trace.append(TraceStep(
                action="LLM(hypothesize)",
                obs=json.dumps(resp, ensure_ascii=False, default=str),
                thought="verbatim fraud-pattern derivation LLM response"))
            logger.info(f"[Explore] Exploration complete ({len(visited)} nodes visited); derived {len(hypotheses)} fraud patterns")
            trace.append(TraceStep(
                action="SynthesizeHypotheses",
                obs=f"{len(hypotheses)} fraud patterns derived from {len(visited)} visited nodes",
                thought="LLM derived fraud patterns, feature names, and evidence chains from traversed paths"))
        elif self.min_hops:
            logger.warning("[Explore] Hypothesis LLM unavailable; skipping target"
                           f" (no fallback in live mode, min_hops={self.min_hops})")
            trace.append(TraceStep(
                action="SkipTarget(LLMUnavailable)",
                obs="hypothesis LLM unavailable (timeout or service failure)",
                thought="no fallback in live mode; return empty hypotheses"))
            return {"hypotheses": [], "trace": trace,
                    "signals": real_signals, "visited": sorted(visited),
                    "effective_subgraph": None, "state": state}
        else:
            hypotheses = self._fallback(subgraph)
            logger.info(f"[Explore] LLM unavailable; deterministically derived {len(hypotheses)} fraud patterns")
            trace.append(TraceStep(action="SynthesizeHypotheses(fallback)",
                                   obs=f"{len(hypotheses)} hypotheses",
                                   thought="LLM unavailable; derived deterministically from structural signals and amounts"))
        raw_hypotheses = hypotheses
        valid, rejected = self._validate_hypotheses(subgraph, hypotheses, strict=from_llm,
                                                    required_hops=required_hops)
        if rejected:
            logger.warning("Hypothesis validation rejections:\n  - " + "\n  - ".join(rejected))
            trace.append(TraceStep(
                action="ValidateHypotheses",
                obs=f"first pass: {len(valid)} accepted / {len(rejected)} rejected",
                thought="check evidence-edge existence, chain contiguity, and depth"))
            hypotheses = valid
            if from_llm and self.llm and raw_hypotheses:
                logger.info(f"[Explore] Hypothesis validation: {len(valid)} accepted / {len(rejected)} rejected; requesting LLM revision")
                plan_req_retry = (", **and the mandatory gremlin_plan for every hypothesis**"
                                  " (steps must anchor with {{SEED}} and cover every evidence_chain hop)"
                                  if self.emit_plan else "")
                retry_user = (hyp_user + "\n\n[VALIDATION FAILED: CORRECT AND RESUBMIT]\n"
                              + "\n".join(rejected)
                              + f"\nRequirements: resubmit valid fraud patterns whose structure is defined by the evidence_chain relation sequence; "
                                f"{chain_req_txt}; segments must be contiguous (each to equals the next from); "
                                f"at least one segment must exclude the center account; use at least three nodes; provide path and {hop_req_txt}; "
                                f"self-contained logic (exact English entity identifiers; no short IDs, aliases, or cross-references); "
                                f"plus feature_name, fraud_pattern, and expected_value_type"
                              + plan_req_retry + ". Return JSON only.")
                resp2 = self.llm.complete_json(hyp_system, retry_user, temperature=0.9)
                if resp2 and isinstance(resp2, dict) and "hypotheses" in resp2:
                    valid2, rejected2 = self._validate_hypotheses(
                        subgraph, resp2["hypotheses"], strict=True,
                        required_hops=required_hops)
                    if valid2:
                        hypotheses = valid2
                        logger.info(f"[Explore] {len(valid2)} hypotheses accepted after revision")
                        trace.append(TraceStep(
                            action="ValidateHypotheses(retry)",
                            obs=f"{len(valid2)} accepted after retry",
                            thought="LLM corrected hypotheses using validation feedback"))
            if not hypotheses:
                lax, _ = self._validate_hypotheses(subgraph, raw_hypotheses, strict=False)
                if lax:
                    logger.warning(f"[Explore] No hypotheses passed strict validation; retaining {len(lax)} with lenient validation")
                    hypotheses = lax
                    trace.append(TraceStep(
                        action="ValidateHypotheses(lax)",
                        obs=f"{len(lax)} retained by lenient validation",
                        thought="no hypothesis passed strict multi-hop validation"))
        else:
            hypotheses = valid
        if not hypotheses and not self.min_hops:
            hypotheses = self._fallback(subgraph)
            trace.append(TraceStep(action="SynthesizeHypotheses(fallback)",
                                   obs=f"{len(hypotheses)} hypotheses",
                                   thought="no hypothesis passed validation; derived deterministically from structural signals and amounts"))
        state.round += 1
        return {"hypotheses": hypotheses, "trace": trace,
                "signals": real_signals, "visited": sorted(visited),
                "effective_subgraph": subgraph.restrict(visited),
                "state": state}

    @staticmethod
    def _fmt_rel(rel: str, rel_name: str) -> str:
        rel = (rel or "").strip()
        rel_name = (rel_name or "").strip()
        if rel and rel_name and rel_name != rel:
            return f"{rel} ({rel_name})"
        return rel or rel_name

    def _frontier(self, subgraph: Subgraph, visited: set[str],
                  per_src: int) -> list[dict]:
        cand: dict[str, dict] = {}
        for nid in visited:
            for n in subgraph.expand_neighbors(nid, None, per_src):
                b = n["nbr"]
                if b in visited:
                    continue
                rec = cand.get(b)
                if rec is None:
                    rec = cand[b] = {
                        "id": b, "srcs": [], "rels": set(),
                        "entity": node_entity(subgraph.nodes.get(b, {})),
                        "hop": subgraph.nodes.get(b, {}).get("first_hop"),
                        "new_deg": sum(1 for e in subgraph.adj.get(b, [])
                                       if e["nbr"] not in visited and e["nbr"] != b),
                        "amt_max": 0.0, "txn": 0,
                    }
                rec["srcs"].append(nid)
                rel = n.get("rel") or ""
                rel_name = n.get("rel_name") or ""
                if rel:
                    rec["rels"].add(self._fmt_rel(rel, rel_name))
                elif rel_name:
                    rec["rels"].add(rel_name)
                amt = n.get("total_amount") or n.get("amt")
                if amt:
                    try:
                        rec["amt_max"] = max(rec["amt_max"], float(amt))
                    except (TypeError, ValueError):
                        pass
                    rec["txn"] += int(n.get("trade_count") or 1)
        out = list(cand.values())
        out.sort(key=lambda r: (-len(r["srcs"]), -r["new_deg"], r["id"]))
        if self.strict_depth:
            base = min((subgraph.nodes[n].get("first_hop") or 0) for n in visited)
            target_hop = base + self._cur_step + 1
            out = [r for r in out if (r["hop"] or 0) == target_hop]
        return out

    def _navigation_view(self, subgraph: Subgraph, state: ExplorerState,
                         quota: int, recent: list[str] | None = None
                         ) -> tuple[str, bool, int]:
        visited = state.visited
        per_src = max(self.top_k, min(64, quota))
        cands = self._frontier(subgraph, visited, per_src)
        has = bool(cands)
        lines = self._visited_subgraph_view(subgraph, state, recent)
        if state.focus and state.focus in subgraph.adj:
            focus_nbrs = {e["nbr"] for e in subgraph.adj.get(state.focus, [])}
            cands = sorted(cands,
                           key=lambda r: (0 if r["id"] in focus_nbrs else 1,
                                          -len(r["srcs"]), -r["new_deg"]))
        show_cap = max(20, min(quota * 3, self.view_cap))
        shown = cands[:show_cap]
        lines.append(f"[EXPANDABLE NEIGHBORS] {len(cands)} candidates; showing {len(shown)}"
                     f" (order: focus first, bridge-source count descending, hub degree descending; from count is within-subgraph sharing; "
                     f"prioritize mediators connected to multiple visited nodes)")
        if not has:
            lines.append("  (none)")
        for r in shown:
            rels = "|".join(sorted(r["rels"])[:3]) or "-"
            amt_txt = (f" amt≈{r['amt_max']:.0f}; txns={r['txn']}"
                       if r.get("amt_max") else "")
            attrs = self._attrs_summary(subgraph, r["id"])
            lines.append(
                f"  - {r['id']} ({r['entity']}) hop={r['hop']} "
                f"new_deg={r['new_deg']} "
                f"from=[{','.join(r['srcs'][:4])}{'…' if len(r['srcs']) > 4 else ''}] "
                f"rel={rels}{amt_txt}"
                + (f" attrs={{{attrs}}}" if attrs else ""))
        if len(cands) > len(shown):
            lines.append(f"  ... {len(cands) - len(shown)} additional candidates omitted (lower bridge/hub priority)")
        return "\n".join(lines), has, len(cands)

    _VIS_EDGE_CAP = 60
    _VIS_ATTR_CAP = 20

    @staticmethod
    def _sid_key(sid: str) -> tuple:
        m = re.match(r"^([A-Za-z]+)(\d+)$", sid or "")
        return (m.group(1), int(m.group(2))) if m else (sid or "", 0)

    @staticmethod
    def _compress_sids(sids: list[str]) -> str:
        items = []
        for sid in sids:
            m = re.match(r"^([A-Za-z]+)(\d+)$", sid or "")
            items.append((m.group(1), int(m.group(2)), sid) if m else (sid or "", 0, sid))
        items.sort()
        out, i = [], 0
        while i < len(items):
            j = i
            while (j + 1 < len(items) and items[j + 1][0] == items[j][0]
                   and items[j + 1][1] == items[j][1] + 1):
                j += 1
            if j - i >= 2:
                out.append(f"{items[i][2]}~{items[j][2]}")
            else:
                out.extend(x[2] for x in items[i:j + 1])
            i = j + 1
        return ", ".join(out)

    def _visited_subgraph_view(self, subgraph: Subgraph, state: ExplorerState,
                               recent: list[str] | None) -> list[str]:
        visited = state.visited
        nodes = subgraph.nodes
        center = subgraph.center
        cent = node_entity(nodes.get(center, {}))
        cattrs = self._attrs_summary(subgraph, center)
        lines = ["[VISITED SUBGRAPH: CUMULATIVE STRUCTURE] (target account/node summary/relations/member attributes; "
                 "src->dst is stored graph direction; amounts are in fen)"]
        lines.append(f"Target account: {center} ({cent}) hop=0"
                     + (f" attrs={{{cattrs}}}" if cattrs else ""))
        if recent:
            lines.append(f"  (added in previous hop: {', '.join(recent[:30])}"
                         + ("..." if len(recent) > 30 else "") + ")")
        groups: dict[str, list[str]] = defaultdict(list)
        for nid in visited:
            groups[node_entity(nodes.get(nid, {}))].append(nid)
        lines.append("Node summary:")
        for ent in sorted(groups.keys(), key=lambda e: (0 if e == cent else 1, e)):
            members = sorted(groups[ent], key=self._sid_key)
            lines.append(f"  - {ent} (members {self._compress_sids(members)}; count={len(members)})")
        agg: dict[tuple, list[str]] = {}
        for e in (getattr(subgraph, "edges", None) or []):
            s, d = e.get("src"), e.get("dst")
            if s not in visited or d not in visited or s == d:
                continue
            raw_rel = e.get("join_field") or e.get("rel") or ""
            rel = (relation_info(raw_rel) or {}).get("en") or raw_rel
            amt = e.get("total_amount") or e.get("amt")
            cnt = e.get("trade_count")
            top = e.get("top_k")
            parts = []
            if amt:
                try:
                    parts.append(f"{float(amt):g}")
                except (TypeError, ValueError):
                    pass
            if cnt:
                parts.append(f"count={int(cnt)}")
            if top not in (None, ""):
                parts.append(f"top_k={top}")
            label = f"{rel} ({', '.join(parts)})" if parts else (rel or "-")
            agg.setdefault((s, d), [])
            if label not in agg[(s, d)]:
                agg[(s, d)].append(label)
        keys = sorted(agg.keys(), key=lambda k: (self._sid_key(k[0]), self._sid_key(k[1])))
        more = max(0, len(keys) - self._VIS_EDGE_CAP)
        lines.append(f"Relations: {len(keys)} directed edge groups"
                     + (f"; showing first {self._VIS_EDGE_CAP}" if more else ""))
        for k in keys[:self._VIS_EDGE_CAP]:
            lines.append(f"  - {k[0]}->{k[1]}: " + "; ".join(agg[k]))
        if not keys:
            lines.append("  (no edges between visited nodes yet)")
        if more:
            lines.append(f"  ... {more} additional edge groups omitted")
        ordered = sorted(visited, key=lambda n: (
            nodes.get(n, {}).get("first_hop")
            if nodes.get(n, {}).get("first_hop") is not None else 99,
            self._sid_key(n)))
        attr_lines = []
        for nid in ordered:
            if nid == center:
                continue
            attrs = self._attrs_summary(subgraph, nid)
            if not attrs:
                continue
            hop = nodes.get(nid, {}).get("first_hop")
            attr_lines.append(
                f"  - {nid} ({node_entity(nodes.get(nid, {}))}) hop={hop}: {{{attrs}}}")
        more_a = max(0, len(attr_lines) - self._VIS_ATTR_CAP)
        lines.append(f"Member attributes (excluding target): {len(attr_lines)} nodes with attributes"
                     + (f"; showing first {self._VIS_ATTR_CAP}" if more_a else ""))
        lines.extend(attr_lines[:self._VIS_ATTR_CAP] or ["  (none)"])
        if more_a:
            lines.append(f"  ... attributes for {more_a} additional nodes omitted")
        return lines

    _ACTION_WHITELIST = {"Expand", "Discard", "Revisit", "Focus", "Stop"}

    def _apply_actions(self, subgraph: Subgraph, state: ExplorerState,
                       actions: list, quota: int, *, step: int, trace: list,
                       recent: list[str], obs_text: str, reason_text: str,
                       nav_notes: list[str]) -> dict:
        visited = state.visited
        frontier_ids = {r["id"] for r in self._frontier(subgraph, visited, 10 ** 6)}
        added_ids: list[str] = []
        used_quota = 0
        discarded_count = 0
        for a in actions:
            if not isinstance(a, dict):
                continue
            act = str(a.get("action", "")).strip()
            if act not in self._ACTION_WHITELIST:
                logger.warning(f"Ignoring unregistered action: {act}")
                continue
            ids = a.get("ids") or []
            if not isinstance(ids, list):
                ids = [ids]
            ids = [str(x).strip() for x in ids if str(x).strip()]
            why = str(a.get("why", "") or "").strip()
            if act == "Expand":
                if self._live():
                    groups_spec = a.get("groups") or []
                    if groups_spec:
                        gids, gnote = self.source.pick_groups(subgraph, groups_spec)
                        ids.extend(gids)
                        if gnote:
                            nav_notes.append(f"[hop {step}: group selection] {gnote}")
                    pend = [x for x in ids if self.source.is_pending(x)]
                    if pend:
                        _ok, rej = self.source.materialize_pending(subgraph, pend)
                        if rej:
                            nav_notes.append(
                                f"[hop {step}: budget] {len(rej)} over-limit candidates were not materialized because this hop budget was exhausted: "
                                f"{','.join(rej[:6])}")
                    if pend or groups_spec:
                        frontier_ids = {r["id"] for r in
                                        self._frontier(subgraph, visited, 10 ** 6)}
                valid = [x for x in ids
                         if x in frontier_ids and x not in visited
                         and x not in state.discarded]
                picked = 0
                for x in valid:
                    if used_quota >= quota:
                        break
                    visited.add(x)
                    added_ids.append(x)
                    frontier_ids.discard(x)
                    used_quota += 1
                    picked += 1
                if picked:
                    state.add_log(step, "Expand", valid[:picked], why)
                    trace.append(TraceStep(action="Expand",
                                           obs=f"added {picked} nodes: {','.join(valid[:picked])}",
                                           thought=why or "expand selected candidates"))
                    if obs_text or reason_text:
                        nav_notes.append(f"[hop {step}: Expand] added {valid[:picked]}: {why}")
            elif act == "Discard":
                if self._live() and a.get("groups"):
                    _n = self.source.discard_groups(a.get("groups") or [])
                    if _n:
                        nav_notes.append(f"[hop {step}: group discard] {_n} over-limit candidates")
                valid_frontier = [x for x in ids if x in frontier_ids and x not in visited]
                valid_visited = [x for x in ids
                                 if x in visited and x != subgraph.center]
                for x in valid_frontier:
                    state.discarded[x] = why or "no investigative value"
                for x in valid_visited:
                    visited.discard(x)
                    state.discarded[x] = why or "retained node judged valueless after deeper inspection"
                valid = valid_frontier + valid_visited
                discarded_count += len(valid_frontier)
                if valid:
                    state.add_log(step, "Discard", valid, why)
                    obs = (f"discarded {len(valid_frontier)} candidates"
                           + (f" + {len(valid_visited)} retained nodes" if valid_visited else ""))
                    trace.append(TraceStep(action="Discard",
                                           obs=f"{obs}: {','.join(valid)}",
                                           thought=why or "judged valueless"))
                    nav_notes.append(f"[hop {step}: Discard] {obs} (reason: {why or 'no investigative value'})")
            elif act == "Revisit":
                valid = [x for x in ids if x in state.discarded and x not in visited]
                for x in valid:
                    if used_quota >= quota:
                        break
                    visited.add(x)
                    added_ids.append(x)
                    state.discarded.pop(x, None)
                    frontier_ids.discard(x)
                    used_quota += 1
                if valid:
                    state.add_log(step, "Revisit", valid[:used_quota], why)
                    trace.append(TraceStep(action="Revisit",
                                           obs=f"restored {len(valid[:used_quota])} discarded nodes: "
                                               f"{','.join(valid[:used_quota])}",
                                           thought=why or "new evidence makes discarded nodes valuable again"))
                    nav_notes.append(f"[hop {step}: Revisit] restored {valid[:used_quota]} (new evidence: {why})")
            elif act == "Focus":
                tid = ids[0] if ids else None
                if tid and tid in visited:
                    state.focus = tid
                    nbrs = [e["nbr"] for e in subgraph.adj.get(tid, [])]
                    nbrs = list(dict.fromkeys(
                        x for x in nbrs
                        if x in frontier_ids and x not in visited
                        and x not in state.discarded))
                    picked_f: list[str] = []
                    for x in nbrs:
                        if used_quota >= quota:
                            break
                        visited.add(x)
                        added_ids.append(x)
                        frontier_ids.discard(x)
                        used_quota += 1
                        picked_f.append(x)
                    state.add_log(step, "Focus", [tid] + picked_f, why)
                    trace.append(TraceStep(action="Focus",
                                           obs=f"focus={tid}; added {len(picked_f)} neighbors: "
                                               f"{','.join(picked_f)}",
                                           thought=why or "add the neighbors of the selected node"))
                    nav_notes.append(f"[hop {step}: Focus] focus={tid}, added neighbors "
                                     f"{picked_f} ({why})")
        recent[:] = added_ids
        return {"added": used_quota, "discarded": discarded_count,
                "topped": 0, "recent": added_ids}

    @staticmethod
    def _plan_contract_block() -> str:
        try:
            from .synthesizer import plan_contract_text
        except Exception as e:
            logger.warning(f"[Explore] Plan contract loading failed ({e}); ablation 2 hypotheses without plans will be rejected")
            return ""
        contract = plan_contract_text()
        return (
            "[ADDITIONAL REQUIREMENT: W/O-SYN produces exploration and plan in one agent call]\n"
            "Every hypothesis must include an executable `gremlin_plan`. It is "
            "part of the feature. There is no later synthesis stage in this mode; a hypothesis without a valid plan is rejected.\n"
            "The plan must cover every evidence_chain hop and follow the injected direction facts. "
            "Anchor the first step with {{SEED}} and follow the contract below exactly.\n\n"
            "The authoritative output JSON schema for this mode is: "
            "Both gremlin_plan and feature_name are mandatory.\n"
            '{"hypotheses":[{"feature_name","feature_name_cn","fraud_pattern","path","hops",'
            '"logic","rationale","expected_value_type","evidence_chain",'
            '"gremlin_plan"}...]}\n'
            'The gremlin_plan shape is:\n'
            '  "gremlin_plan": {"steps": [{"id": "s1", "desc": "...", '
            '"gremlin": "g.V().hasId({{SEED}})", "output": "id_list"}, ...], '
            '"final": {"type": "count", "step": "sN"}, "business_rationale": "...", '
            '"business_explanation": "2-4 plain-English sentences derived from fraud_pattern and rationale"}\n\n'
            + contract)

    @staticmethod
    def _plan_output_block() -> str:
        return ("[PRE-SUBMISSION CHECKLIST FOR W/O-SYN]\n"
                "1) Does every hypothesis include gremlin_plan? Feature name and plan are mandatory together; "
                "a hypothesis without a plan is rejected.\n"
                "2) gremlin_plan must be a JSON object, not a string; steps must be nonempty and "
                "cover every evidence_chain hop in order.\n"
                "3) Anchor the first step with g.V().hasId({{SEED}}), reference upstream steps with {{STEP:sX}}, "
                "and follow the system contract for edge filters, direction, and numeric types.\n"
                "4) final must reference the last step and its type must match expected_value_type.\n"
                "Complete hypothesis example:\n"
                '  {"feature_name": "shared_bank_card_account_cnt", "hops": 3,\n'
                '   "evidence_chain": [{"from": "A001", "rel": "wechat_pay_account_bind_bank_card", "to": "O001", "hop": 1}, ...],\n'
                '   "logic": "...",\n'
                '   "gremlin_plan": {"steps": [{"id": "s1", "gremlin": "g.V().hasId({{SEED}})", "output": "id_list"},\n'
                '                              {"id": "s2", "gremlin": "g.V().hasId(within({{STEP:s1}})) ... '
                'per-hop edge filters and direction follow the contract ... .dedup().id().toList()", "output": "id_list"}],\n'
                '                    "final": {"type": "count", "step": "s2"}}}\n')

    @staticmethod
    def _skill_block_for(stage: str, skill_memory: object | None) -> str:
        if skill_memory is None or not hasattr(skill_memory, "playbook_text"):
            return ""
        try:
            return skill_memory.playbook_text(stage) or ""
        except Exception as e:
            logger.warning(f"[Explore] Skill block injection failed ({e}); skipping block")
            return ""

    def _global_review(self, subgraph: Subgraph, state: ExplorerState,
                       trace: list, nav_notes: list[str]) -> None:
        visited = state.visited
        if len(visited) < 4:
            logger.info("[Review] Too few visited nodes (<4); skipping global review")
            return
        raw_frontier = self._frontier(subgraph, visited, 10 ** 6)
        dist = self._visited_bfs(subgraph, visited)
        cand_depth: dict[str, int] = {}
        for r in raw_frontier:
            ds = [dist[s] for s in r["srcs"] if s in dist]
            if ds:
                cand_depth[r["id"]] = min(ds) + 1
        over_hops = [r["id"] for r in raw_frontier
                     if r["id"] not in state.discarded
                     and cand_depth.get(r["id"], 0) > self.max_hops]
        frontier = [r for r in raw_frontier
                    if r["id"] not in state.discarded
                    and cand_depth.get(r["id"], 0) <= self.max_hops][:20]
        if over_hops:
            logger.info(f"[Review] Excluded {len(over_hops)} frontier candidates that would exceed "
                        f"max_hops={self.max_hops}: {','.join(over_hops[:6])}")
        frontier_ids = {r["id"] for r in frontier}
        add_cap = max(8, len(visited) // 5)
        max_depth = self._visited_depth(subgraph, visited)
        user = _REVIEW_USER_TMPL.format(
            center=subgraph.center,
            center_entity=node_entity(subgraph.nodes.get(subgraph.center, {})),
            visited_n=len(visited), max_depth=max_depth, add_cap=add_cap,
            structure_summary=self._structure_summary(subgraph, visited),
            visited_view=self._review_visited_view(subgraph, visited),
            frontier_view=self._review_frontier_view(subgraph, frontier),
            discarded_block=self._render_discarded_block(state, max_items=20)
            or "(discard pool is empty)")
        logger.info(f"[Review] Global review: {len(visited)} nodes visited, maximum depth {max_depth}, "
                    f"{len(frontier)} frontier candidates; requesting LLM decisions (discard/revisit/expand)")
        resp = self.llm.complete_json(_REVIEW_SYSTEM, user, temperature=0.6)
        if not isinstance(resp, dict):
            logger.warning("[Review] Global-review decision parsing failed; preserving hop-by-hop exploration results")
            return
        review_text = str(resp.get("review", "") or "").strip()
        actions = resp.get("actions") or []
        if not isinstance(actions, list):
            actions = []
        trace.append(TraceStep(
            action="LLM(global_review)",
            obs=json.dumps(resp, ensure_ascii=False, default=str),
            thought=review_text or "verbatim global-review decision"))
        dropped: list[str] = []
        revived: list[str] = []
        added: list[str] = []
        depth_blocked: list[str] = []
        for a in actions:
            if not isinstance(a, dict):
                continue
            act = str(a.get("action", "")).strip()
            if act not in ("Discard", "Revisit", "Expand"):
                logger.warning(f"[Review] Ignoring unregistered action: {act}")
                continue
            ids = a.get("ids") or []
            if not isinstance(ids, list):
                ids = [ids]
            ids = [str(x).strip() for x in ids if str(x).strip()]
            why = str(a.get("why", "") or "").strip()
            if act == "Discard":
                for x in ids:
                    if x in visited and x != subgraph.center:
                        visited.discard(x)
                        state.discarded[x] = why or "global review: no contribution to the risk structure"
                        dropped.append(x)
            elif act == "Revisit":
                for x in ids:
                    if len(revived) + len(added) >= add_cap:
                        break
                    if x in state.discarded and x not in visited:
                        if cand_depth.get(x, 0) > self.max_hops:
                            depth_blocked.append(x)
                            continue
                        visited.add(x)
                        state.discarded.pop(x, None)
                        revived.append(x)
            elif act == "Expand":
                for x in ids:
                    if len(revived) + len(added) >= add_cap:
                        break
                    if x in frontier_ids and x not in visited \
                            and x not in state.discarded:
                        visited.add(x)
                        added.append(x)
        if dropped or revived or added:
            state.add_log(max_depth, "GlobalReview",
                          {"dropped": dropped, "revived": revived, "added": added},
                          review_text)
        obs = (f"dropped={len(dropped)} ({','.join(dropped[:8])})" if dropped else "dropped=0")
        obs += (f"; revived={len(revived)} ({','.join(revived[:8])})" if revived else "; revived=0")
        obs += (f"; added={len(added)} ({','.join(added[:8])})" if added else "; added=0")
        logger.info(f"[Review] Final review: {obs} -> {len(visited)} retained nodes")
        if added or revived:
            hist: dict = {}
            for x in added + revived:
                d = cand_depth.get(x, "?")
                hist[d] = hist.get(d, 0) + 1
            logger.info(f"[Review] Added-node depth distribution (visited BFS, max_hops="
                        f"{self.max_hops}): {hist} -> maximum depth after review "
                        f"{self._visited_depth(subgraph, visited)} hops")
        if depth_blocked:
            logger.info(f"[Review] Rejected {len(depth_blocked)} revisits that would exceed "
                        f"max_hops={self.max_hops}: {','.join(depth_blocked[:6])}")
        if review_text:
            logger.info(f"[Review] Global assessment: {review_text}")
        trace.append(TraceStep(action="GlobalReview", obs=obs,
                               thought=review_text or "global review: final node membership decisions"))
        nav_notes.append(f"[GlobalReview] {obs} ({review_text})")

    def _review_visited_view(self, subgraph: Subgraph, visited: set[str],
                             cap: int = 150) -> str:
        rows = []
        for nid in visited:
            nd = subgraph.nodes.get(nid, {})
            hop = nd.get("first_hop")
            inner_deg = sum(1 for e in subgraph.adj.get(nid, []) if e["nbr"] in visited)
            amt = 0.0
            for e in subgraph.adj.get(nid, []):
                if e["nbr"] in visited:
                    try:
                        amt += float(e.get("total_amount") or e.get("amt") or 0)
                    except (TypeError, ValueError):
                        pass
            rows.append((hop if hop is not None else 99, -inner_deg, nid,
                         node_entity(nd), inner_deg, amt))
        rows.sort()
        lines = []
        for hop, _neg_deg, nid, ent, inner_deg, amt in rows[:cap]:
            amt_txt = f" amt≈{amt:.0f}" if amt else ""
            lines.append(f"  - {nid} ({ent}) hop={hop} internal_degree={inner_deg}{amt_txt}"
                         + (" [center]" if nid == subgraph.center else ""))
        if len(rows) > cap:
            lines.append(f"  ... {len(rows) - cap} additional visited nodes omitted")
        return "\n".join(lines) or "  (empty)"

    @staticmethod
    def _review_frontier_view(subgraph: Subgraph, frontier: list[dict]) -> str:
        if not frontier:
            return "  (no unexpanded frontier candidates)"
        lines = []
        for r in frontier:
            rels = "|".join(sorted(r["rels"])[:3]) or "-"
            amt_txt = (f" amt≈{r['amt_max']:.0f}; txns={r['txn']}"
                       if r.get("amt_max") else "")
            lines.append(
                f"  - {r['id']} ({r['entity']}) hop={r['hop']} "
                f"new_deg={r['new_deg']} "
                f"from=[{','.join(r['srcs'][:4])}{'…' if len(r['srcs']) > 4 else ''}] "
                f"rel={rels}{amt_txt}")
        return "\n".join(lines)

    @staticmethod
    def _render_discarded_block(state: ExplorerState, max_items: int = 8) -> str:
        if not state.discarded:
            return ""
        lines = ["[DISCARDED: Revisit is allowed only with new evidence]"]
        for i, (nid, reason) in enumerate(state.discarded.items()):
            if i >= max_items:
                break
            new_info = state.action_log and any(
                "Expand" == lg.get("action") and nid in lg.get("ids", [])
                for lg in state.action_log)
            tail = " (later Expand actions may provide new evidence)" if new_info else ""
            lines.append(f"  - {nid} discard reason: {reason or 'none'}{tail}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _render_online_feedback_block(feedback: str) -> str:
        if not feedback:
            return ""
        return ("\n[ONLINE EVALUATOR GUIDANCE FROM THE PREVIOUS ROUND ON THIS SUBGRAPH]\n"
                + "\n".join(f"  - {x}" for x in feedback.strip().splitlines() if x.strip())
                + "\n  Requirement: for positive high/mid signals, deepen variants of the same relation family; for negative low signals, "
                  "avoid that feature family/path and choose an unexplored branch or a stronger structural constraint.\n")

    def _structure_summary(self, subgraph: Subgraph, visited: set[str]) -> str:
        if not visited:
            return "(empty)"
        ents = Counter(node_entity(subgraph.nodes.get(n, {})) for n in visited)
        hops = Counter(subgraph.nodes.get(n, {}).get("first_hop") for n in visited)
        rels: Counter = Counter()
        for n in visited:
            for e in subgraph.adj.get(n, []):
                if e["nbr"] in visited and (e.get("rel") or e.get("rel_name")):
                    rels[self._fmt_rel(e.get("rel"), e.get("rel_name"))] += 1
        lines = [
            f"  Entity distribution: {', '.join(f'{k}×{v}' for k, v in ents.most_common())}",
            "  Hop distribution: " + ", ".join(
                f"{'?' if k is None else k} hops x {v}"
                for k, v in sorted(hops.items(), key=lambda x: (x[0] is None, x[0]))),
            "  Top internal relations: " + (", ".join(f"{k}×{v}" for k, v in rels.most_common(8))
                                or "(none)"),
        ]
        media = []
        for n in visited:
            nd = subgraph.nodes.get(n, {})
            if is_account(nd):
                continue
            accts = [e["nbr"] for e in subgraph.adj.get(n, [])
                     if is_account(subgraph.nodes.get(e["nbr"], {}))]
            if len(accts) >= 2:
                media.append((len(accts), n, node_entity(nd)))
        media.sort(key=lambda x: -x[0])
        if media:
            lines.append("  Mediators shared by multiple accounts (within-subgraph sharing; assess strength using entity type and "
                         "beware of extremely high-degree public hubs): " + "; ".join(
                             f"{n}({ent})-> {c} accounts in subgraph" for c, n, ent in media[:8]))
        try:
            amt = subgraph.amount_observations()
            if amt.get("has_amount"):
                lines.append(
                    f"  Center amount observations: outgoing {amt['out_sum']:.0f} / incoming {amt['in_sum']:.0f} / "
                    f"max single {amt['max_single']:.0f} / txns={amt['count']} / "
                    f"counterparties={len(amt['counterparties'])}")
        except Exception:
            pass
        return "\n".join(lines)

    def _traversal_view(self, subgraph: Subgraph, visited: set[str]) -> str:
        order = sorted(visited, key=lambda n: (
            subgraph.nodes.get(n, {}).get("first_hop") if
            subgraph.nodes.get(n, {}).get("first_hop") is not None else 99, n))
        shown = order[:self.node_cap]
        lines = []
        for nid in shown:
            nd = subgraph.nodes.get(nid, {})
            ent = node_entity(nd)
            attrs = self._attrs_summary(subgraph, nid)
            lines.append(f"[{nid}] ({ent}) hop={nd.get('first_hop')}"
                         + (f" attrs={{{attrs}}}" if attrs else ""))
            nbrs = subgraph.expand_neighbors(nid, None, self.nbr_cap)
            total = len(subgraph.adj.get(nid, []))
            if not nbrs:
                lines.append("   (no neighbors)")
            src_set = {(e.get("src"), e.get("dst"), e.get("join_field") or e.get("rel"))
                       for e in (getattr(subgraph, "edges", None) or [])}
            for n in nbrs:
                nid2 = n["nbr"]
                ent2 = node_entity(subgraph.nodes.get(nid2, {}))
                mark = "visited" if nid2 in visited else "unexpanded"
                amt = n.get("total_amount") or n.get("amt")
                amt_txt = ""
                if amt:
                    try:
                        amt_txt = f" amt={float(amt):.0f}"
                    except (TypeError, ValueError):
                        pass
                jf = n.get("rel")
                if (nid, nid2, jf) in src_set:
                    dir_txt = " | outgoing(out)"
                elif (nid2, nid, jf) in src_set:
                    dir_txt = " | incoming(in)"
                else:
                    dir_txt = ""
                lines.append(f"   - {nid2} ({ent2}) rel={self._fmt_rel(n.get('rel'), n.get('rel_name'))}"
                             f"{dir_txt}{amt_txt} [{mark}]")
            if total > len(nbrs):
                lines.append(f"   ... {total - len(nbrs)} additional adjacencies not shown")
        if len(order) > len(shown):
            lines.append(f"... {len(order) - len(shown)} additional visited nodes not expanded "
                         f"(see the structure summary above)")
        return "\n".join(lines)

    def _commit_picks(self, subgraph: Subgraph, visited: set[str],
                      picks: list[Any], quota: int) -> tuple[int, list[str]]:
        added_ids: list[str] = []
        frontier = {r["id"] for r in self._frontier(subgraph, visited, 10 ** 6)}
        for p in picks:
            if not isinstance(p, str):
                continue
            p = p.strip()
            if not p or p in visited:
                continue
            if p not in frontier:
                logger.warning(f"Ignoring invented or unreachable LLM node: {p}")
                continue
            visited.add(p)
            added_ids.append(p)
            if len(added_ids) >= quota:
                break
        return len(added_ids), added_ids

    def _topup(self, subgraph: Subgraph, visited: set[str],
               n: int, exclude: set[str] | None = None) -> tuple[int, list[str]]:
        if n <= 0:
            return 0, []
        exclude = exclude or set()
        added: list[str] = []
        for r in self._frontier(subgraph, visited, 10 ** 6):
            if len(added) >= n:
                break
            if r["id"] in exclude:
                continue
            visited.add(r["id"])
            added.append(r["id"])
        return len(added), added

    def _live(self) -> bool:
        return self.source is not None and getattr(self.source, "live", False)

    def explore_random(self, subgraph: Subgraph) -> Subgraph:
        src = getattr(self, "source", None)
        if src is not None:
            src.prepare(subgraph, None)
            src.finalize(subgraph, set(subgraph.nodes.keys()))
        return subgraph

    @staticmethod
    def _visited_bfs(subgraph: Subgraph, visited: set[str]) -> dict:
        from collections import deque
        center = subgraph.center
        if center not in visited:
            return {}
        dist = {center: 0}
        q = deque([center])
        while q:
            u = q.popleft()
            for e in subgraph.adj.get(u, []):
                v = e["nbr"]
                if v in visited and v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def _visited_depth(self, subgraph: Subgraph, visited: set[str]) -> int:
        return max(self._visited_bfs(subgraph, visited).values(), default=0)

    def _auto_deepen(self, subgraph: Subgraph, state: ExplorerState,
                     visited: set[str], trace: list) -> int:
        total = 0
        for _ in range(self.min_hops + 2):
            dist = self._visited_bfs(subgraph, visited)
            depth = max(dist.values(), default=0)
            if depth >= self.min_hops:
                break
            frontier = self._frontier(subgraph, visited, 10 ** 6)
            if not frontier:
                break
            deep_srcs = {n for n, d in dist.items() if d == depth}
            extend = [r for r in frontier if deep_srcs & set(r["srcs"])]
            pool = extend or frontier
            pool.sort(key=lambda r: (-len(r["srcs"]), -r["new_deg"], r["id"]))
            picks = [r["id"] for r in pool[:5]]
            if not picks:
                break
            for x in picks:
                visited.add(x)
            if self._live():
                self.source.ensure_expanded(subgraph, picks)
            total += len(picks)
        if total:
            trace.append(TraceStep(action="AutoDeepen", obs=f"+{total}",
                                   thought=f"depth gate: extended to at least {self.min_hops} hops"))
        return total

    @staticmethod
    def _edge_exists(subgraph: Subgraph, a: str, b: str, rel: str | None) -> bool:
        if a not in subgraph.nodes or b not in subgraph.nodes:
            return False
        for e in subgraph.adj.get(a, []):
            if e["nbr"] == b and (rel is None or e["rel"] is None
                                  or rel in e["rel"] or e["rel"] in rel):
                return True
        return False

    def _segment_edge(self, subgraph: Subgraph, src_set: set,
                      a: str, b: str, rel: str = "", direction: str = ""):
        """Resolve one evidence segment against relation and stored direction, not endpoints alone."""
        requested = (relation_info(rel) or {}).get("en") or rel
        for edge in subgraph.adj.get(a, []):
            if edge.get("nbr") != b:
                continue
            actual = edge.get("rel_id") or edge.get("rel")
            if requested and actual != requested:
                continue
            stored = self._seg_direction(src_set, a, b, actual)
            if direction in ("out", "in") and stored != direction:
                continue
            return edge, stored
        return None, ""

    def _validate_hypotheses(self, subgraph: Subgraph, hypotheses: list[dict],
                             strict: bool = False,
                             required_hops: int = 0) -> tuple[list[dict], list[str]]:
        center = subgraph.center
        valid, rejected = [], []
        src_set = set()
        for edge in (getattr(subgraph, "edges", None) or []):
            raw_rel = edge.get("join_field") or edge.get("rel")
            src_set.add((edge.get("src"), edge.get("dst"), raw_rel))
            public_rel = (relation_info(raw_rel) or {}).get("en") or raw_rel
            src_set.add((edge.get("src"), edge.get("dst"), public_rel))
        for h in hypotheses:
            name = h.get("name") or h.get("feature_name", "?")
            plan_norm: dict | None = None
            if self.emit_plan:
                plan_raw = h.get("gremlin_plan")
                plan_key = "gremlin_plan"
                if plan_raw is None:
                    plan_raw, plan_key = h.get("plan"), "plan"
                if plan_raw is None:
                    rejected.append(f"Hypothesis '{name}' has no gremlin_plan "
                                    f"(the query must be produced together with the hypothesis); rejected")
                    continue
                if isinstance(plan_raw, str):
                    rejected.append(f"Hypothesis '{name}': {plan_key} is a string, not a JSON object "
                                    f"(provide an object with steps/final); rejected")
                    continue
                if not isinstance(plan_raw, dict):
                    rejected.append(f"Hypothesis '{name}': {plan_key} has type "
                                    f"{type(plan_raw).__name__}, expected an object; rejected")
                    continue
                if not (plan_raw.get("steps") or []):
                    rejected.append(f"Hypothesis '{name}': {plan_key}.steps is missing or empty "
                                    f"(steps must cover every evidence_chain hop); rejected")
                    continue
                plan_norm = normalize_plan(plan_raw)
                ok, verr = validate_plan(plan_norm)
                if not ok:
                    rejected.append(f"Hypothesis '{name}': gremlin_plan failed syntactic validation: {verr}; rejected")
                    continue
                plan_norm["business_explanation"] = str(
                    plan_raw.get("business_explanation")
                    or h.get("business_explanation") or "").strip()
            evt = (h.get("expected_value_type") or "").strip().lower()
            if strict and not str(h.get("logic") or "").strip():
                rejected.append(f"Hypothesis '{name}' lacks self-contained logic; rejected")
                continue
            if strict and evt not in {"count", "amount", "ratio", "flag"}:
                rejected.append(f"Hypothesis '{name}' has missing or invalid expected_value_type; rejected")
                continue
            chain = h.get("evidence_chain") or []
            evs: list[dict] = []
            chain_err = ""
            if chain:
                for i, seg in enumerate(chain):
                    if not isinstance(seg, dict):
                        chain_err = f"segment {i+1} is not an object"
                        break
                    a, b = seg.get("from"), seg.get("to")
                    if not a or not b:
                        chain_err = f"segment {i+1} lacks from/to"
                        break
                    edge, stored = self._segment_edge(
                        subgraph, src_set, a, b, str(seg.get("rel") or ""),
                        str(seg.get("direction") or ""))
                    if edge is None:
                        chain_err = (f"segment {i+1} {a}->{b} with rel={seg.get('rel') or '?'} "
                                     f"and direction={seg.get('direction') or 'auto'} is not a real edge")
                        break
                    if i > 0 and evs[-1]["b"] != a:
                        chain_err = (f"chain is not contiguous: segment {i} ends at {evs[-1]['b']} "
                                     f"but segment {i+1} starts at {a}")
                        break

                    evs.append({"a": a, "b": b,
                                "rel": edge.get("rel_id") or edge.get("rel"),
                                "rel_name": edge.get("rel_name"),
                                "direction": stored,
                                "hop": seg.get("hop"), "note": seg.get("note")})
            else:
                evs = []
                for e in (h.get("evidence") or []):
                    if not (isinstance(e, dict) and e.get("a") and e.get("b")):
                        continue
                    edge, stored = self._segment_edge(
                        subgraph, src_set, e["a"], e["b"], str(e.get("rel") or ""),
                        str(e.get("direction") or ""))
                    if edge is None:
                        continue
                    evs.append({"a": e["a"], "b": e["b"],
                                "rel": edge.get("rel_id") or edge.get("rel"),
                                "rel_name": edge.get("rel_name"),
                                "direction": stored,
                                "hop": e.get("hop"), "note": e.get("note", "")})
            try:
                declared_hops = int(h.get("hops") or 0)
                for i, edge in enumerate(evs):
                    edge["hop"] = int(edge.get("hop") or i + 1)
            except (TypeError, ValueError):
                rejected.append(f"Hypothesis '{name}' has a non-integer hops or segment hop value; rejected")
                continue
            if self.min_hops and declared_hops < self.min_hops:
                rejected.append(f"Hypothesis '{name}' declares hops={declared_hops or '?'} < "
                                f"minimum depth {self.min_hops}; rejected")
                continue
            has_chain = bool(chain)
            if chain_err:
                rejected.append(f"Hypothesis '{name}' has an invalid evidence chain: {chain_err}; rejected")
                continue
            if has_chain and not evs:
                rejected.append(f"Hypothesis '{name}' lacks valid real-edge evidence; rejected")
                continue
            if not has_chain:
                if strict or evt != "amount":
                    rejected.append(f"Hypothesis '{name}' lacks a valid evidence_chain with real edges; rejected")
                    continue
            skip_multihop = (evt == "amount")
            if strict and not skip_multihop:
                if len(evs) < 2:
                    rejected.append(f"Hypothesis '{name}' has only {len(evs)} valid evidence edges; "
                                    f"at least 2 real edges are required for a multi-hop structure; rejected")
                    continue
                nodes_involved = {x for e in evs for x in (e["a"], e["b"])}
                if len(nodes_involved) < 3:
                    rejected.append(f"Hypothesis '{name}' involves only {len(nodes_involved)} nodes; "
                                    f"at least 3 are required; rejected")
                    continue
                if not any(e["a"] != center and e["b"] != center for e in evs):
                    rejected.append(f"Hypothesis '{name}': every evidence edge touches the center {center} "
                                    f"(trivial one-hop structure); at least one edge must exclude the center; rejected")
                    continue
                if required_hops >= 2:
                    try:
                        h_hops = int(h.get("hops") or 0)
                    except (TypeError, ValueError):
                        h_hops = 0
                    if self.min_hops:
                        if h_hops > required_hops:
                            rejected.append(f"Hypothesis '{name}' declares hops={h_hops} > observed maximum depth "
                                            f"{required_hops}; rejected")
                            continue
                        if len(evs) < h_hops:
                            rejected.append(f"Hypothesis '{name}' has only {len(evs)} real evidence edges < "
                                            f"declared hops={h_hops} (the chain must reach the declared depth); rejected")
                            continue
                    else:
                        if h_hops != required_hops:
                            rejected.append(f"Hypothesis '{name}' declares hops={h_hops or '?'}, which differs from "
                                            f"the observed maximum depth {required_hops}; rejected")
                            continue
                        if len(evs) < required_hops:
                            rejected.append(f"Hypothesis '{name}' has only {len(evs)} real evidence edges and does not "
                                            f"reach hop {required_hops} (at least {required_hops} segments required); rejected")
                            continue

            norm = dict(h)
            norm["feature_name"] = h.get("feature_name") or h.get("name") or "unnamed"
            norm["rationale"] = (h.get("rationale") or h.get("intuition")
                                 or h.get("discrimination") or "")
            norm["intuition"] = norm["rationale"]
            norm["fraud_pattern"] = (h.get("fraud_pattern") or h.get("risk_pattern")
                                     or norm["rationale"] or "")
            norm.pop("risk_pattern", None)
            if not norm.get("hops"):
                norm["hops"] = len(evs)

            _cn = (h.get("feature_name_cn") or h.get("name_cn") or "").strip()
            if not _cn and norm["fraud_pattern"]:
                _cn = norm["fraud_pattern"].split(",")[0].strip()[:20]
            norm["feature_name_cn"] = _cn
            norm["evidence_chain"] = [
                {"from": e["a"], "rel": e.get("rel"), "to": e["b"],
                 "direction": e.get("direction") or "",
                 "hop": e.get("hop", i + 1), "note": e.get("note", "")}
                for i, e in enumerate(evs)]
            if plan_norm is not None:
                norm["gremlin_plan"] = plan_norm
            else:
                norm.pop("gremlin_plan", None)
                norm.pop("plan", None)
            dir_block = Explorer._direction_block(subgraph, evs)
            if dir_block:
                _llm_logic = (norm.get("logic") or "").strip()
                if _llm_logic and re.search(r"\b(outgoing|incoming)\b", _llm_logic, re.I):
                    logger.warning(f"[Explore] Hypothesis '{name}' logic still contains outgoing/incoming wording "
                                   f"(prohibited by requirement 10); injected direction facts are authoritative")
                norm["logic"] = dir_block + ("\n" + _llm_logic if _llm_logic else "")
            valid.append(norm)
        return valid, rejected

    @staticmethod
    def _seg_direction(src_set: set, a: str, b: str, jf: str) -> str:
        if not jf:
            return ""
        if (a, b, jf) in src_set:
            return "out"
        if (b, a, jf) in src_set:
            return "in"
        return ""

    @staticmethod
    def _direction_block(subgraph: Subgraph, evs: list[dict]) -> str:
        if not evs:
            return ""
        lines = []
        prev = "target account"
        for i, e in enumerate(evs):
            rel = e.get("rel") or "?"
            b_ent = (node_entity(subgraph.nodes.get(e["b"], {}))
                     if subgraph is not None else "") or "unknown entity"
            hop = e.get("hop", i + 1)
            d_txt = {"out": "outgoing", "in": "incoming"}.get(e.get("direction"), "unknown direction")
            lines.append(f"  hop {hop}: {prev} --follow `{rel}` {d_txt}--> {b_ent}")
            prev = f"hop-{hop} {b_ent}"
        return ("[CHAIN DIRECTION: injected from stored subgraph direction and authoritative for comparison]\n"
                + "\n".join(lines))

    @staticmethod
    def _real_signal_block(real_signals: list[dict]) -> str:
        if not real_signals:
            return "(no salient structure detected yet)"
        lines = []
        for m in real_signals:
            mt = m["signal"]
            if mt == "shared_strong_field":
                lines.append(f"  - shared_strong_field: mediator {m.get('medium')} "
                             f"accounts={m.get('accounts')} (shared by {m.get('group_size')} accounts in subgraph)")
            elif mt == "fan_out":
                lines.append(f"  - fan_out: {m.get('center')} -> {m.get('count')} downstream "
                             f"{m.get('downstream')}")
            elif mt == "mutual_transfer":
                lines.append(f"  - mutual_transfer: {m.get('a')} <-> {m.get('b')}")
            else:
                lines.append(f"  - {mt}: {m}")
        return "\n".join(lines)

    @staticmethod
    def _signal_hint(real_signals: list[dict]) -> str:
        types = sorted({m["signal"] for m in real_signals})
        return ", ".join(types) if types else "(none detected yet; explore by bridging, entity type, and amount)"

    @staticmethod
    def _attrs_summary(subgraph: Subgraph, nid: str, max_attrs: int = 4,
                       val_cap: int = 18) -> str:
        nd = subgraph.nodes.get(nid, {})
        attrs = nd.get("attributes") or {}
        picked: list[tuple[int, str]] = []
        for k, v in attrs.items():
            kl = str(k).lower()
            if any(p in kl for p in Explorer._PII_ATTR_SUBSTR):
                continue
            if v is None:
                continue
            vs = str(v).strip()
            if not vs or vs.lower() == "none":
                continue
            if len(vs) > val_cap:
                vs = vs[:val_cap] + "…"
            picked.append((len(vs), f"{k}={vs}"))
        picked.sort(key=lambda x: x[0])
        return ", ".join(v for _, v in picked[:max_attrs])

    @staticmethod
    def _fallback(subgraph: Subgraph) -> list[dict]:
        signals = subgraph.detect_signals()
        amt = subgraph.amount_observations(subgraph.center)
        out = []
        for m in signals:
            if m["signal"] == "shared_strong_field":
                medium = m.get("medium")
                accts = m.get("accounts") or []
                chain = ([{"from": accts[0], "rel": None, "to": medium, "hop": 1,
                           "note": "account bound to a strong identifier"}] if (medium and accts) else [])
                rp = (f"Small group sharing {m['field']} ({m['group_size']} accounts in the subgraph); "
                      f"a shared strong identifier suggests a coordinated group")
                out.append({
                    "feature_name": f"{m['field']}_shared_smallgroup",
                    "feature_name_cn": f"Small group sharing {m['field']}",
                    "fraud_pattern": rp,
                    "evidence_chain": chain,
                    "expected_value_type": "flag",
                    "rationale": "Normal accounts rarely share strong identifiers with many others; "
                                 "reuse is markedly higher around fraud accounts",
                })
            elif m["signal"] == "fan_out":
                ds = m.get("downstream") or []
                chain = ([{"from": m["center"], "rel": None, "to": ds[0], "hop": 1,
                           "note": "transfer from the target to its first downstream account"}] if ds else [])
                out.append({
                    "feature_name": "fan_out_transfer",
                    "feature_name_cn": "Fan-out transfers",
                    "fraud_pattern": f"The target transfers to {m['count']} downstream accounts, "
                                    f"suggesting fund dispersal",
                    "evidence_chain": chain,
                    "expected_value_type": "count",
                    "rationale": "Dispersal hubs have many more downstream counterparties than normal accounts",
                })
            elif m["signal"] == "mutual_transfer":
                chain = [{"from": m["a"], "rel": None, "to": m["b"], "hop": 1,
                          "note": "bidirectional fund flow"}]
                out.append({
                    "feature_name": "mutual_transfer_pair",
                    "feature_name_cn": "Mutual transfer pair",
                    "fraud_pattern": "Two accounts transfer funds to each other, suggesting fund cycling",
                    "evidence_chain": chain,
                    "expected_value_type": "flag",
                    "rationale": "Normal counterparties rarely transfer in both directions",
                })
        if amt.get("has_amount"):
            if amt["out_sum"] > 0:
                out.append({
                    "feature_name": "total_out_amount",
                    "feature_name_cn": "Total outgoing amount",
                    "fraud_pattern": "Large outgoing totals may indicate cash-out or fraudulent outflow",
                    "evidence_chain": [],
                    "expected_value_type": "amount",
                    "rationale": "Fraud accounts tend to move larger outgoing volumes than normal accounts",
                })
            if amt["max_single"] >= 100000:
                out.append({
                    "feature_name": "single_large_transfer",
                    "feature_name_cn": "Single large transfer",
                    "fraud_pattern": "A single very large transfer may indicate laundering or fund pooling",
                    "evidence_chain": [],
                    "expected_value_type": "amount",
                    "rationale": "Normal accounts rarely make single very large transfers",
                })
        if not out:
            out.append({"feature_name": "bank_card_shared",
                        "feature_name_cn": "Shared bank card",
                        "fraud_pattern": "A shared bank card may link accounts under coordinated control",
                        "evidence_chain": [],
                        "expected_value_type": "flag",
                        "rationale": "Bank-card sharing across accounts is a testable coordination signal"})
        for h in out:
            h["intuition"] = h["rationale"]
        return out
