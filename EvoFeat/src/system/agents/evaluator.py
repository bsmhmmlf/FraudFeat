from __future__ import annotations

import json

from ..llm.client import LLMClient
from ..utils import get_logger
from ..core.graph import node_entity
from ..core.schema import Candidate, IVTier, MemoryExample, TraceStep
from ..memory.retriever import make_retriever


logger = get_logger("evaluator")
_SYSTEM = """You are the IV Evaluator for graph risk-control features. Given a candidate feature (name, computation logic, query, business explanation) and its target-centered source subgraph, estimate its information value without executing queries or accessing a database.
Return JSON only: {"iv":number,"conf":0.0~1.0,"tier":"low|medium|high","reason":"short justification"}.
iv is a non-negative predicted IV under the classical unbounded WoE definition. Calibration anchors: <0.02 no predictive power; 0.02-0.1 weak; 0.1-0.3 moderate; 0.3-0.5 strong; >0.5 very strong.
conf in [0,1] expresses how reliable your estimate is. tier is your overall assessment, guided by the intervals low=[0,0.2), medium=[0.2,0.5), high=[0.5,inf).
Use the subgraph to check that the evidence chain exists and matches the observable entity, sharing, centrality, attribute, amount, and count signals. It is one fraud-centered local view, not population evidence. Base the estimate primarily on the measured IV of the retrieved similar features.
Lower conf and iv when the claimed chain is absent or the query does not implement the stated logic, chain, or output type.
Evidence-chain arrows from->to indicate chain organization, not storage direction. Each segment's outgoing/incoming annotation is authoritative: outgoing means from is src and requires outE().inV(); incoming means from is dst and requires inE().outV().
Cross-check business explanation, logic, and query. Penalize contradictions and vague explanations.
Each retrieved example carries the measured IV, the earlier prediction, and a calibration tag. Correct in the opposite direction of over- or under-calibrated analogues; well-calibrated analogues can be used directly."""

_USER_TMPL = """[RETRIEVED SIMILAR FEATURES WITH MEASURED IV]
{memory}

[SOURCE SUBGRAPH: fraud center={center}, {n_nodes} nodes, {n_edges} edges]
{subgraph}

[CANDIDATE]
Feature name: {name}{name_cn}
Expected value type: {evt}
Computation logic:
{logic}
Evidence chain:
{chain}
Query:
{dsl}
Risk rationale: {rationale}
Business explanation:
{explain}

Predict iv, conf, and tier. Return JSON only."""


def _tier_to_iv(tier: IVTier) -> float:
    return {"high": 0.7, "mid": 0.35, "low": 0.1}[tier.value]


def _clamp_iv(v: float) -> float:
    return round(max(0.0, v), 4)


def _parse_tier(v) -> IVTier:
    s = str(v or "").strip().lower()
    if s in ("medium", "mid", "moderate"):
        return IVTier.MID
    if s == "high":
        return IVTier.HIGH
    return IVTier.LOW


class Evaluator:
    def __init__(self, llm: LLMClient | None = None, top_k: int = 4,
                 retriever_mode: str = "hybrid",
                 eval_skill_memory=None):
        self.llm = llm or LLMClient()
        self.top_k = top_k
        self.retriever = make_retriever(retriever_mode, llm=llm)
        self.eval_skill_memory = eval_skill_memory

    def _skill_block(self) -> str:
        sm = self.eval_skill_memory
        if sm is None:
            return ""
        try:
            txt = str(sm.playbook_text() or "").strip()
        except Exception as e:
            logger.warning(f"[Evaluate] Calibration playbook injection failed ({e}); skipping block")
            return ""
        if not txt:
            return ""
        return ("[EVALUATOR SKILL: CALIBRATION GUIDANCE]\n"
                "Use it to calibrate IV and confidence; individual retrieved examples override it when they conflict.\n"
                + txt)

    def warm_memory(self, memory: list[MemoryExample]) -> int:
        warm = getattr(self.retriever, "warm_items", None)
        if not callable(warm):
            return 0
        try:
            return warm(memory, key="feature_desc")
        except Exception as e:
            logger.warning(f"[Evaluate] Memory warmup failed ({e}); computing on demand during retrieval")
            return 0

    @staticmethod
    def describe(c: Candidate) -> str:
        return (f"{c.feature_name} | {c.value_type} | "
                f"{c.dsl_program if isinstance(c.dsl_program, str) else str(c.dsl_program)}")

    def predict_iv(self, cand: Candidate, memory: list[MemoryExample],
                   subgraph=None) -> tuple[IVTier, float, float, str] | None:
        logger.info(f"[Evaluate] Predicting IV tier for '{cand.feature_name}': retrieve top-{self.top_k} memories"
                    f" ({type(self.retriever).__name__}) + LLM analogy (no graph query)")

        sims = self.retriever.retrieve(cand.feature_name, memory, k=self.top_k,
                                       key="feature_desc")
        if sims and isinstance(sims[0], str):
            mem_block = "\n\n".join(sims[:self.top_k]) or "(empty memory)"
        else:
            mem_block = "\n\n".join(m.to_prompt() for m in sims[:self.top_k]) or "(empty memory)"
        dsl_text = cand.dsl_program if isinstance(cand.dsl_program, str) else str(cand.dsl_program)
        cn = getattr(cand, "feature_name_cn", "") or ""
        name_cn = f" ({cn})" if cn else ""
        explain = str(getattr(cand, "business_explanation", "") or "").strip() or "(not provided)"
        user = _USER_TMPL.format(memory=mem_block, name=cand.feature_name,
                                 name_cn=name_cn, dsl=dsl_text, rationale=cand.rationale,
                                 explain=explain,
                                 evt=getattr(cand, "expected_value_type", "") or "(not declared)",
                                 logic=(getattr(cand, "logic", "") or "")[:800] or "(not provided)",
                                 chain=self._chain_view(cand),
                                 center=getattr(subgraph, "center", "?"),
                                 n_nodes=len(subgraph.nodes) if subgraph is not None else 0,
                                 n_edges=len(subgraph.edges) if subgraph is not None else 0,
                                 subgraph=self._subgraph_view(subgraph))
        skill_block = self._skill_block()
        if skill_block:
            user = skill_block + "\n\n" + user
            if not getattr(self, "_skill_block_logged", False):
                self._skill_block_logged = True
                logger.info(f"[Evaluate] Calibration playbook injected ({len(skill_block)} characters of family priors, "
                            f"before individual few-shot examples; individual examples override conflicts)")
        resp = self.llm.complete_json(_SYSTEM, user, temperature=0.3)
        if resp and isinstance(resp, dict) and ("iv" in resp or "tier" in resp):
            tier = _parse_tier(resp.get("tier"))
            try:
                conf = min(1.0, max(0.0, float(resp.get("conf", 0.5))))
            except (TypeError, ValueError):
                conf = 0.5
            reason = str(resp.get("reason", "") or "")
            cand.trace.append(TraceStep(
                action="LLM(evaluate)",
                obs=json.dumps(resp, ensure_ascii=False, default=str),
                thought=reason or "verbatim IV-evaluation LLM response"))
            try:
                iv = _clamp_iv(float(resp.get("iv", _tier_to_iv(tier))))
            except (TypeError, ValueError):
                iv = _tier_to_iv(tier)
            cand.predicted_iv = iv
            logger.info(f"[Evaluate] Prediction: tier={tier.value} conf={conf:.2f} iv={iv:.3f}"
                        f" | reason: {reason[:80]}")
            return tier, conf, iv, reason
        cand.predicted_iv = None
        logger.info("[Evaluate] LLM unavailable; no prediction for this candidate")
        return None

    _PII_ATTR_SUBSTR = ("cre_id", "id_card", "cert", "card_no", "card",
                        "phone", "mobile", "password", "pwd", "token")

    @staticmethod
    def _chain_view(cand) -> str:
        chain = getattr(cand, "evidence_chain", None) or []
        if not chain:
            return "  (no evidence chain; evaluate from query text and memory analogies)"
        lines = ["  (from->to only organizes the chain; |outgoing/|incoming marks stored direction)"]
        for i, s in enumerate(chain):
            if not isinstance(s, dict):
                continue
            rel = s.get("rel") or s.get("rel_name") or s.get("join_field") or "?"
            note = str(s.get("note") or "")[:40]

            d = {"out": "outgoing", "in": "incoming"}.get(s.get("direction"), "")
            d_txt = f" |{d}" if d else ""
            lines.append(f"  {i+1}. {s.get('from', '?')} -[{rel}]-> {s.get('to', '?')}"
                         f" (hop={s.get('hop', i+1)}{d_txt})" + (f": {note}" if note else ""))
        return "\n".join(lines) or "  (no evidence chain; evaluate from query text and memory analogies)"

    @classmethod
    def _subgraph_view(cls, subgraph, node_cap: int = 40, nbr_cap: int = 10,
                       max_attrs: int = 4, val_cap: int = 18) -> str:
        if subgraph is None:
            return "(no subgraph view; evaluate from query text and memory analogies)"
        nodes = subgraph.nodes
        order = sorted(nodes, key=lambda n: (
            nodes[n].get("first_hop") if nodes[n].get("first_hop") is not None else 99, n))
        shown = order[:node_cap]
        lines: list[str] = []
        for nid in shown:
            nd = nodes.get(nid, {})
            ent = node_entity(nd)
            mark = " (center)" if nid == subgraph.center else ""
            attrs = cls._attrs_brief(nd, max_attrs, val_cap)
            lines.append(f"[{nid}] ({ent}) hop={nd.get('first_hop')}{mark}"
                         + (f" attrs={{{attrs}}}" if attrs else ""))
            adj = subgraph.adj.get(nid, [])
            for e in adj[:nbr_cap]:
                n2 = e["nbr"]
                ent2 = node_entity(nodes.get(n2, {}))
                amt = e.get("total_amount") or e.get("amt")
                amt_txt = ""
                if amt:
                    try:
                        amt_txt = f" amt={float(amt):.0f} fen"
                    except (TypeError, ValueError):
                        pass
                lines.append(f"   - {n2} ({ent2}) rel={e.get('rel_name') or e.get('rel')}"
                             f"{amt_txt}")
            if len(adj) > nbr_cap:
                lines.append(f"   ... {len(adj) - nbr_cap} additional adjacencies omitted")
        if len(order) > len(shown):
            lines.append(f"... {len(order) - len(shown)} additional nodes omitted in ascending hop order")
        return "\n".join(lines)

    @classmethod
    def _attrs_brief(cls, nd: dict, max_attrs: int, val_cap: int) -> str:
        picked: list[tuple[int, str]] = []
        for k, v in (nd.get("attributes") or {}).items():
            kl = str(k).lower()
            if any(p in kl for p in cls._PII_ATTR_SUBSTR):
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
