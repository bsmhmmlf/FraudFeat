from __future__ import annotations

import json
from typing import Any

from ..utils import get_logger
from ..core.schema import Candidate, IVTier
from ..evolution import eval_material as em
from ..evolution.reward import evaluate_batch
from ..evolution.evaluator_eval import EvaluatorMemory
from ..evolution.skill_memory import SkillMemory
from ..evolution.thresholds import ThresholdPolicy
from .gating import Gate


logger = get_logger("offline_phase")


def run_offline(*, queue: list[Candidate], pool: list[Candidate],
                eval_memory: EvaluatorMemory, skill_memory: SkillMemory,
                gate: Gate, m_episodes: int = 2,
                episode_index: int = 0, force_budget: bool = False,
                threshold_policy: ThresholdPolicy | None = None,
                process_negatives: list[Candidate] | None = None,
                synthesis_memory=None,
                diversity_tracker=None,
                no_evaluator: bool = False,
                no_evolution: bool = False,
                no_diversity_guide: bool = False,
                eval_skill_memory=None, executor=None) -> dict:
    policy = threshold_policy or ThresholdPolicy()
    info: dict[str, Any] = {"grounded": 0, "entered_pool": [],
                            "new_skills": 0, "new_synth_skills": 0,
                            "pos_samples": 0, "neg_samples": 0}
    if no_evaluator:
        selected = gate.select_random(queue, gate.cfg.budget_per_episode,
                                      seed=episode_index)
    else:
        selected = gate.select_stratified(queue, gate.cfg.budget_per_episode)
    if not selected:
        logger.info("[Offline] h) Active top-K selection: no candidates; skipping")
        return info
    info["grounded"] = len(selected)
    logger.info(f"[Offline] h) Active top-K selection (stratified): {len(selected)} candidates selected"
                f" (budget {gate.cfg.budget_per_episode})")
    selected_ids = {id(c) for c in selected}
    queue[:] = [c for c in queue if id(c) not in selected_ids]
    info["queue_remaining"] = len(queue)
    logger.info(f"[Offline] {len(queue)} unselected candidates retained for the next episode; "
                "selected candidates leave the queue after this measurement attempt")
    logger.info("[Offline] i) Batch ComputeTrueIV (ground-truth evaluation, the only expensive step)")
    evaluate_batch(selected, pool, executor=executor)
    missing = [c for c in selected if c.true_iv is None]
    if missing:
        logger.warning(
            f"True IV missing for {len(missing)}/{len(selected)} candidates: this episode skips them "
            f"for evaluator memory, templates, skills, and pool admission; they remain frozen. "
            f"If the live graph is connected, check the ComputeTrueIV execution path.")
    if no_evaluator or no_evolution:
        added_mem = 0
        logger.info(f"[Offline] j) {'w/o-Eval' if no_evaluator else 'Base-Agent (no memory)'}"
                    ": skipping evaluator memory updates")
    else:
        added_mem = eval_memory.absorb(selected)
        logger.info(f"[Offline] j) Evaluator memory updated: +{added_mem} entries ({len(eval_memory)} total)")
    if eval_skill_memory is not None and not no_evaluator and not no_evolution:
        written = list(getattr(eval_memory, "_last_written", None) or [])
        if not written:
            logger.info("[Offline] j2) No IV_Rag entries written this episode; skipping calibration playbook with no LLM call")
        else:
            material = em.build_material(written, eval_memory.memory,
                                         episode=episode_index)
            diff = eval_skill_memory.summarize(
                material, episode=episode_index,
                forbidden_names={str(m.doc_name or "") for m in eval_memory.memory})
            if diff:
                info["eval_skill_changed"] = True
                info["eval_skill_diff"] = {
                    k: list(diff.get(k) or []) for k in ("added", "revised", "removed")}
            else:
                info["eval_skill_changed"] = False
    gaps = [abs(c.true_iv - c.predicted_iv) for c in selected
            if c.true_iv is not None and c.predicted_iv is not None]
    if gaps:
        info["avg_iv_gap"] = round(sum(gaps) / len(gaps), 4)
        tier_gaps: dict[str, list[float]] = {}
        for c in selected:
            if c.true_iv is None or c.predicted_iv is None:
                continue
            t = IVTier.from_value(c.true_iv).value
            tier_gaps.setdefault(t, []).append(abs(c.true_iv - c.predicted_iv))
        info["iv_gap_by_tier"] = {t: round(sum(v) / len(v), 4)
                                  for t, v in sorted(tier_gaps.items())}
        by_tier = " ".join(f"{t}={sum(v)/len(v):.3f}(n={len(v)})"
                           for t, v in sorted(tier_gaps.items()))
        logger.info(f"[Offline] j+) Evaluator calibration: mean predicted-vs-true |ΔIV|={info['avg_iv_gap']:.3f}"
                    f" (by tier: {by_tier}; {len(gaps)} IV pairs stored in memory "
                    f"for subsequent few-shot calibration)")
    graded = ([] if no_evaluator else
              [c for c in selected
               if c.true_iv is not None and getattr(c, "predicted_tier", None) is not None])
    if graded:
        def _acc(rows):
            return round(sum(1 for c in rows
                             if c.predicted_tier == IVTier.from_value(c.true_iv))
                         / len(rows), 4)
        by_true: dict[str, list] = {}
        for c in graded:
            by_true.setdefault(IVTier.from_value(c.true_iv).value, []).append(c)
        signed = [c.predicted_iv - c.true_iv for c in graded
                  if c.predicted_iv is not None]
        info["episode_stats"] = {
            "episode": episode_index,
            "n_grounded": len(graded),
            "tier_acc": _acc(graded),
            "tier_acc_by_tier": {t: {"acc": _acc(rows), "n": len(rows)}
                                 for t, rows in sorted(by_true.items())},
            "avg_abs_div": (round(sum(abs(c.predicted_iv - c.true_iv)
                                      for c in graded if c.predicted_iv is not None)
                                  / len(signed), 4) if signed else None),
            "avg_signed_div": (round(sum(signed) / len(signed), 4)
                               if signed else None),
        }
        logger.info(f"[Offline] j++) Episode {episode_index} evaluator performance: "
                    f"tier accuracy={info['episode_stats']['tier_acc']:.2f}"
                    f" (n={len(graded)}), "
                    f"signed ΔIV={info['episode_stats']['avg_signed_div']}")
    for c in selected:
        if c.true_iv is None:
            continue
        pool.append(c)
        info["entered_pool"].append(c.feature_name)
    logger.info(f"[Offline] k) Feature pool updated: admitted {info['entered_pool']} (pool size {len(pool)})")
    if no_evolution:
        info["skill_evolved"] = True
        logger.info("[Offline] m) Base-Agent (no memory): skipping offline evolution (playbooks, digests, and failure diagnosis)")
    if episode_index % m_episodes == 0 and not no_evolution:
        info["skill_evolved"] = True
        episodes = []
        skipped_frozen = 0
        for c in selected:
            if c.true_iv is None:
                skipped_frozen += 1
                continue
            ep = _explore_episode_full(c)
            ep["iv"] = c.true_iv if c.true_iv is not None else 0.0
            ep["redundancy"] = c.redundancy
            episodes.append(ep)
        if skipped_frozen:
            logger.info(f"[Offline] m) Skipping {skipped_frozen} candidates without true IV in exploration playbooks"
                        f" (frozen, not failed; no positive/negative labeling without ground truth)")
        labeled = policy.label_samples(episodes)
        episodes = labeled
        info["pos_samples"] = sum(1 for e in episodes if e.get("positive"))
        info["neg_samples"] = sum(1 for e in episodes if e.get("negative"))
        logger.info(f"[Offline] m) Exploration playbook material: {len(episodes)} candidates with true IV"
                    f" (positive {info['pos_samples']} / negative {info['neg_samples']}, labeled by true-IV thresholds; "
                    f"synthesis failures go to m+, not exploration playbooks)")
        new_skills = skill_memory.summarize(episodes, pool=pool)
        info["new_skills"] = len(new_skills)
        logger.info(f"[Offline] m) Skill evolution: positive {info['pos_samples']} / negative "
                    f"{info['neg_samples']}, {len(new_skills)} playbooks changed"
                    f" (paradigm, navigation, and hypothesis playbooks revised independently)")
        gov = skill_memory.govern()
        info["skills_retired"] = len(gov["retired"])
        info["skills_merged"] = len(gov["merged"])
        skill_memory.save()
        if synthesis_memory is not None:
            synth_eps_all = _synthesis_episodes(process_negatives)
            synth_eps = [e for e in synth_eps_all
                         if e.get("negative") or e.get("repaired_pass")]
            new_ss = synthesis_memory.summarize(synth_eps)
            info["new_synth_skills"] = len(new_ss)
            n_neg = sum(1 for e in synth_eps if e.get("negative"))
            n_rep = sum(1 for e in synth_eps if e.get("repaired_pass"))
            n_pos = sum(1 for e in synth_eps_all if e.get("positive"))
            logger.info(f"[Offline] m+) Synthesis skill evolution: {len(synth_eps)} learning episodes"
                        f" (terminal failures {n_neg} + repaired passes {n_rep}; "
                        f"{n_pos - n_rep} first-pass successes are monitored only), "
                        f"derived {len(new_ss)} synthesis skills")
            gov_s = synthesis_memory.govern()
            if gov_s["retired"] or gov_s["merged"]:
                synthesis_memory.save()
    if diversity_tracker is not None:
        sat = diversity_tracker.update_from_pool(pool, episode=episode_index)
        if sat:
            info["saturated_paradigms"] = sorted(sat)

    from ..evolution import explore_guidance as eg
    info["guidance"] = eg.build_guidance_block(
        selected=selected, pool=pool, tracker=diversity_tracker,
        enable_coverage=not no_diversity_guide)
    return info


def _plain(v):
    try:
        return json.loads(json.dumps(v, ensure_ascii=False, default=str))
    except Exception:
        try:
            return str(v)
        except Exception:
            return "(non-serializable field omitted)"


def _render_chain(c) -> str:
    """Render the hypothesis evidence chain for synthesis episodes (hop + direction)."""
    chain = getattr(c, "evidence_chain", None) or []
    lines = []
    for i, s in enumerate(chain):
        if not isinstance(s, dict):
            continue
        rel = s.get("rel") or s.get("rel_name") or s.get("join_field") or "?"
        note = str(s.get("note") or "")[:40]
        d = {"out": "outgoing", "in": "incoming"}.get(s.get("direction"), "")
        d_txt = f" | {d}" if d else ""
        lines.append(f"  {i+1}. {s.get('from', '?')} -[{rel}]-> {s.get('to', '?')}"
                     f" (hop={s.get('hop', i + 1)}{d_txt})" + (f": {note}" if note else ""))
    return "\n".join(lines)


def _explore_episode_full(c: Candidate) -> dict:
    chain = getattr(c, "evidence_chain", None) or []
    hop_max = 0
    for s in chain:
        if isinstance(s, dict):
            try:
                hop_max = max(hop_max, int(s.get("hop") or 0))
            except (TypeError, ValueError):
                pass
    tier = getattr(c, "predicted_tier", None)
    return {
        "predicted_iv": c.predicted_iv,
        "predicted_conf": c.predicted_conf,
        "true_iv": c.true_iv,
        "positive": None,
        "negative": None,
        "iv": None, "redundancy": None,
        "feature": c.feature_name,
        "feature_cn": getattr(c, "feature_name_cn", "") or "",
        "value_type": c.value_type,
        "expected_value_type": getattr(c, "expected_value_type", "") or "",
        "hops": hop_max or None,
        "fraud_pattern": getattr(c, "fraud_pattern", "") or "",
        "rationale": c.rationale or "",
        "logic": getattr(c, "logic", "") or "",
        "evidence_chain": _plain(chain),
        "synth_ok": bool(getattr(c, "synth_ok", True)),
        "business_explanation": getattr(c, "business_explanation", "") or "",
        "predicted_tier": str(getattr(tier, "value", "")),
        "predicted_reason": getattr(c, "predicted_reason", "") or "",
        "consistency_report": getattr(c, "consistency_report", "") or "",
        "consistency_detail": _plain(getattr(c, "consistency_detail", None) or {}),
        "exec_error": getattr(c, "exec_error", "") or "",
        "exec_attempts": int(getattr(c, "exec_attempts", 0) or 0),
        "feature_value": repr(getattr(c, "feature_value", None)),
        "execution_trace": _plain(getattr(c, "execution_trace", None) or []),
        "err_history": _plain(getattr(c, "_err_history", None) or []),
        "failed_plan": _plain(getattr(c, "_failed_plan", None) or {}),
        "skills_used": list(getattr(c, "_skills_used", None) or []),
        "trace": [t.to_dict() for t in (c.trace or [])],
    }


def _synthesis_episode(c: Candidate) -> dict | None:
    trace = getattr(c, "trace", None) or []
    repairs = [t for t in trace if str(getattr(t, "action", "")).startswith("ReAct")]
    fallback = bool(getattr(c, "_synth_fallback", False)) or \
        any("fallback" in str(getattr(t, "action", "")).lower() for t in trace)
    exec_attempts = int(getattr(c, "exec_attempts", 0) or 0)
    exec_mode = exec_attempts > 0
    value = getattr(c, "feature_value", None)
    exec_error = getattr(c, "exec_error", "") or ""
    detail = getattr(c, "consistency_detail", None) or {}
    verdict = str(detail.get("consistency") or "").strip()
    inconsistent = bool(verdict) and verdict != "consistent"
    exec_failed = exec_mode and value is None and bool(exec_error)

    if exec_mode:
        passed = (not fallback) and (value is not None) and not exec_failed \
            and not inconsistent
        negative = fallback or exec_failed or inconsistent
    else:
        report = str(getattr(c, "consistency_report", "") or "")
        failed_prefixes = (
            "synthesis failed", "semantic validation failed", "validation failed")
        passed = bool(report) and not report.startswith(failed_prefixes) and not fallback
        negative = fallback
    if not trace and not passed and not negative:
        return None
    consistency_txt = ""
    if verdict:
        consistency_txt = (f"{verdict} | {detail.get('inconsistency_category', '')}: "
                           f"{detail.get('inconsistency_reason', '')}")[:300]
    ep = {
        "feature": c.feature_name,
        "value_type": getattr(c, "value_type", ""),
        "hops": len(getattr(c, "evidence_chain", None) or []),
        "expected_value_type": getattr(c, "expected_value_type", ""),
        "logic": (getattr(c, "logic", "") or "")[:800],
        "chain_view": _render_chain(c),
        "vocab": getattr(c, "_synth_vocab", None) or {},
        "synth_trace": [dict(t) for t in (getattr(c, "_synth_trace", None) or [])],
        "consistency_full": _plain(getattr(c, "consistency_detail", None) or {}),
        "err_history": [str(x) for x in (getattr(c, "_err_history", None) or [])],
        "failed_plan": getattr(c, "_failed_plan", None),
        "passed": passed,
        "first_try": passed and not repairs,
        "repaired_pass": passed and bool(repairs),
        "fallback": fallback,
        "repairs": len(repairs),
        "errors": [str(t.obs) for t in repairs][:8],
        "skills_used": list(getattr(c, "_skills_used", []) or []),
        "true_iv": c.true_iv,
        "positive": passed,
        "negative": negative,
        "exec_mode": exec_mode,
        "exec_value": value,
        "exec_error": exec_error[:300] if exec_error else "",
        "inconsistent": inconsistent,
        "consistency": consistency_txt,
        "explorer_fault": bool(getattr(c, "_explorer_fault", False)),
    }
    return ep


def _synthesis_episodes(process_negatives: list | None) -> list[dict]:
    """Build Synthesizer updates exclusively from pending failure set N_t.

    N_t includes terminal failures and validation failures later repaired within
    the budget. A clean measured candidate is not Synthesizer update material.
    """
    episodes = []
    seen = set()
    pending = [c for c in (process_negatives or [])
               if not getattr(c, "_n_consumed", False)]
    for c in pending:
        if id(c) in seen:
            continue
        seen.add(id(c))
        ep = _synthesis_episode(c)
        if ep:
            episodes.append(ep)
    return episodes
