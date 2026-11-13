from __future__ import annotations

from ..utils import get_logger
from ..parallel import run_parallel
from ..core.graph import Subgraph
from ..core.schema import Candidate, IVTier
from ..agents.explorer import Explorer
from ..agents.synthesizer import Synthesizer
from ..agents.evaluator import Evaluator
from ..evolution.evaluator_eval import EvaluatorMemory
from ..evolution.skill_memory import SkillMemory


logger = get_logger("online_phase")


def process_subgraph(subgraph: Subgraph, *, explorer: Explorer,
                     synthesizer: Synthesizer, evaluator: Evaluator,
                     eval_memory: EvaluatorMemory, skill_memory: SkillMemory,
                     pool: list[Candidate], queue: list[Candidate],
                     synthesis_memory=None,
                     diversity_tracker=None,
                     episode_index: int = 0,
                     max_workers: int = 1,
                     guidance_block: str = "",
                     online_feedback: str = "",
                     no_evaluator: bool = False,
                     no_explorer: bool = False,
                     no_synthesizer: bool = False) -> tuple[list[Candidate], str]:
    skills = skill_memory.list_all()
    paradigm_block = skill_memory.playbook_text("paradigm")
    exp = explorer.explore(subgraph, skills=skills,
                           online_feedback=(online_feedback or None),
                           skill_memory=skill_memory,
                           guidance_block=guidance_block,
                           paradigm_block=paradigm_block,
                           random_nodes=no_explorer)
    hypotheses = exp["hypotheses"]
    if not hypotheses:
        return [], ""
    base_trace = exp["trace"]
    produced: list[Candidate] = []
    synth_view = exp.get("effective_subgraph") or subgraph
    synth_skills = (synthesis_memory.list_all("synthesize")
                    if synthesis_memory is not None else None)

    def _synthesize(hypo: dict, i: int) -> Candidate:
        return synthesizer.synthesize(hypo, synth_view, pool,
                                      list(base_trace), index=i,
                                      total=len(hypotheses),
                                      skills=synth_skills,
                                      skill_memory=synthesis_memory,
                                      preset_plan=(hypo.get("gremlin_plan")
                                                   if no_synthesizer else None))

    cands = [c for c in run_parallel(
        hypotheses, _synthesize, max_workers=max_workers,
        stage=f"Synthesize {len(hypotheses)} feature hypotheses") if c is not None]
    for cand in cands:
        cand.episode = episode_index
    to_eval: list[Candidate] = []
    for cand in cands:
        if cand.synth_ok:
            to_eval.append(cand)
        else:
            cand._negative = True
            produced.append(cand)
    if no_evaluator:
        for cand in to_eval:
            queue.append(cand)
            produced.append(cand)
        return produced, ""

    def _predict(cand: Candidate, i: int):
        return evaluator.predict_iv(cand, eval_memory.memory, subgraph=synth_view)

    preds = run_parallel(to_eval, _predict, max_workers=max_workers,
                         stage=f"Evaluate IV for {len(to_eval)} candidate features")

    round_ivs: list[float] = []
    for cand, pred in zip(to_eval, preds):
        if pred is None:
            tier, conf, iv, reason = IVTier.LOW, 0.0, None, ""
        else:
            tier, conf, iv, reason = pred
        cand.predicted_tier = tier
        cand.predicted_conf = conf
        cand.predicted_iv = iv
        cand.predicted_reason = reason
        if iv is not None:
            round_ivs.append(float(iv))
        queue.append(cand)
        produced.append(cand)

    steer = _build_online_feedback(cands, to_eval, preds, 0)
    logger.info(f"[Online] Subgraph {subgraph.center} complete: {len(cands)} candidates, "
                f"maximum predicted IV={max(round_ivs, default=0.0):.3f}; all queued "
                f"(no online pruning), steer {'generated' if steer else 'empty'} for the next mining round")
    return produced, steer


def _build_online_feedback(cands: list[Candidate], to_eval: list[Candidate],
                           preds: list, rnd: int) -> str:
    if not preds:
        return ""
    lines: list[str] = []
    for cand, pred in zip(to_eval, preds):
        if pred is None:
            continue
        tier, conf, iv, reason = pred
        if iv is None:
            continue
        iv = float(iv)
        conf = float(conf)
        basis = (reason or cand.rationale or "")[:80]
        if tier in (IVTier.HIGH, IVTier.MID):
            if conf >= 0.7:
                lines.append(
                    f"'{cand.feature_name}' predicted iv={iv:.3f} (tier={tier.value}); "
                    f"high-confidence positive assessment (conf={conf:.2f}); basis: {basis}. "
                    f"Continue deepening variants of this relation family.")
            elif conf < 0.6:
                lines.append(
                    f"'{cand.feature_name}' predicted iv={iv:.3f} (tier={tier.value}); "
                    f"positive but uncertain assessment (conf={conf:.2f}); basis: {basis}. "
                    f"Keep the direction but add stronger structural constraints or rarer mediators.")
            else:
                lines.append(
                    f"'{cand.feature_name}' predicted iv={iv:.3f} (tier={tier.value}, conf={conf:.2f}); "
                    f"basis: {basis}. Keep this direction and consider related features.")
        elif conf >= 0.7:
            lines.append(
                f"'{cand.feature_name}' predicted iv={iv:.3f} (tier={tier.value}); "
                f"high-confidence low assessment (conf={conf:.2f}); basis: {basis}. "
                f"Avoid this family and move to an unexplored branch.")
        elif conf < 0.6:
            lines.append(
                f"'{cand.feature_name}' predicted iv={iv:.3f} (tier={tier.value}); "
                f"low-confidence low assessment (conf={conf:.2f}, evaluator blind spot); basis: {basis}. "
                f"The candidate remains pending true measurement. Retry once with stronger constraints or rarer mediators, or choose another branch.")
        else:
            lines.append(
                f"'{cand.feature_name}' predicted iv={iv:.3f} (tier={tier.value}) was rated low; "
                f"basis: {basis}. Avoid similar features and choose an unexplored branch or stronger structural constraints.")
    if not lines:
        return ""
    return "\n".join(lines[:5])
