from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from ..utils import get_logger, load_config, ensure_dir
from ..llm.client import LLMClient
from ..llm.rate_limiter import configure_rate_limiter
from ..core.schema import Candidate, IVTier, TraceStep
from ..agents.explorer import Explorer
from ..agents.synthesizer import Synthesizer
from ..agents.evaluator import Evaluator
from ..evolution.evaluator_eval import EvaluatorMemory, validate_static_assets
from ..evolution.eval_skill_memory import EvalSkillMemory
from ..evolution.skill_memory import SkillMemory
from ..evolution.synthesis_skill_memory import SynthesisSkillMemory
from ..evolution.diversity import ParadigmDiversityTracker
from ..evolution import coldstart
from ..evolution.thresholds import ThresholdPolicy
from ..data.loader import SubgraphStream
from ..execution import build_executor_from_config, preflight
from ..evolution.family import structure_signature
from ..evolution.failure_taxonomy import error_signature
from ..selfcheck import verify as verify_deps
from .online_phase import process_subgraph
from .offline_phase import run_offline
from .gating import Gate, GateConfig
from ..paths import PROJECT_ROOT, SYSTEM_ROOT


logger = get_logger("loop")
_OFFLINE_DIVIDER = "=" * 64


_WORKSPACE = PROJECT_ROOT

_DISK_MIN_FREE_MB = 500


def _disk_guard(out_dir: Path) -> None:
    try:
        free_mb = shutil.disk_usage(str(out_dir)).free / (1024 * 1024)
    except OSError:
        return
    if free_mb < _DISK_MIN_FREE_MB:
        raise SystemExit(f"[Disk guard] Free space {free_mb:.0f}MB < {_DISK_MIN_FREE_MB}MB; "
                         f"aborting explicitly. Free disk space and resume the run")


def _load_eval_seeds(path: str | Path) -> list[tuple[str, float]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Cold-start seed file must be a JSON array: {path}")
    seeds: list[tuple[str, float]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Seed entry {i+1} is not an object: {item!r}")
        desc = str(item.get("desc") or "").strip()
        if not desc:
            name = str(item.get("feature_name") or "").strip()
            if not name:
                raise ValueError(f"Seed entry {i+1} lacks desc/feature_name: {item!r}")
            desc = (f"Feature name: {name}\n"
                    f"Structure: {item.get('structure', '')}\n"
                    f"Rationale: {item.get('rationale', '')}")
        seeds.append((desc, float(item.get("iv", 0.0))))
    return seeds


def eval_skill_mode_of(*, no_evaluator: bool, no_evolution: bool,
                       static_skills: str | Path | None) -> str:
    if no_evaluator:
        return "off"
    if static_skills:
        return "frozen"
    if no_evolution:
        return "off"
    return "live"


def synthesizer_kwargs_of(*, no_synthesizer: bool, max_repair: int,
                          semantic_check: bool) -> dict:
    if not no_synthesizer:
        return {"max_repair": max_repair, "semantic_check": semantic_check}
    return {"max_repair": 0, "semantic_check": True}


def run(*, num_episodes: int = 3, episode_size: int = 2,
        m_episodes: int = 2,
        max_subgraphs: int | None = None, verbose: bool = True,
        max_repair: int = 3, max_hops: int = 5,
        node_budgets: list[int] | None = None, explore_topup: bool = False,
        strict_depth: bool = False, global_review: bool = True,
        skills_path: str | Path | None = None, persist_skills: bool = True,
        reset_skills: bool = False,
        synth_skills_path: str | Path | None = None, persist_synth: bool = True,
        reset_synth: bool = False,
        eval_memory_path: str | Path | None = None, persist_eval: bool = True,
        reset_eval: bool = False,
        eval_seeds_path: str | Path | None = None,
        input_path: str | Path | None = None,
        qpm: int | None = None, max_concurrency: int | None = None,
        threshold_mode: str = "fixed",
        retriever_mode: str = "hybrid",
        semantic_check: bool = True,
        eg_exec: bool = False,
        explore_live: bool = False,
        data_dir: str | Path | None = None,
        offline_panel: str | Path | None = None,
        targets_path: str | Path | None = None,
        account_type: str = "individual",
        target_sample: int = 100,
        random_seed: int = 20260825,
        per_hop_caps="50,250,500,1000,2000",
        cold_start: bool = False, cold_start_reset: bool = False,
        enrich_skills: bool = False,
        run_id: str | None = None,
        no_explorer: bool = False, no_synthesizer: bool = False,
        no_evaluator: bool = False, no_diversity_guide: bool = False,
        no_evolution: bool = False,
        static_skills: str | Path | None = None,
        valid_target: int | None = None,
        max_episodes: int = 20,
        episode_resume: bool = False) -> dict:
    verify_deps("agent run")
    max_hops = int(max_hops)
    if static_skills:
        validate_static_assets(static_skills)
    if run_id:
        out_dir = ensure_dir(_WORKSPACE / "runs" / run_id)
        os.environ["IV_RESULTS_DIR"] = str(out_dir / "iv_values")
        mem_dir = ensure_dir(out_dir / "memory")
        _resume = (out_dir / "progress.json").exists() or any(mem_dir.iterdir())
        if _resume and (reset_skills or reset_synth or reset_eval):
            logger.info(f"[Run isolation] Existing artifacts for run_id={run_id}; resuming, "
                        f"skipping reset flags to preserve this run's memory and RAG")
            reset_skills = reset_synth = reset_eval = False
        skills_path = skills_path or (mem_dir / "skills.json")
        synth_skills_path = synth_skills_path or (mem_dir / "synth_skills.json")
        eval_memory_path = eval_memory_path or str(mem_dir / "eval_memory.json")
        logger.info(f"[Run isolation] run_id={run_id}, all artifacts under {out_dir}"
                    f" ({'resume' if _resume else 'fresh start'})")
    else:
        out_dir = None

    _frozen_read = False
    if static_skills:
        fz = Path(static_skills)
        if not fz.is_dir():
            raise SystemExit(f"[Static-Agent] Frozen asset directory does not exist: {fz}"
                             f" (generate it with experiments/static_agent_materialize.py first)")
        skills_path = fz
        synth_skills_path = fz / "synthesis_playbook.md"
        eval_memory_path = fz / "eval_memory.json"
        persist_skills = persist_synth = persist_eval = False
        no_evolution = True
        _frozen_read = True
        logger.info(f"[Static-Agent] Loaded frozen prewritten memory: {fz} (no updates; "
                    f"retriever mode={retriever_mode})")
    elif no_evolution:
        retriever_mode = "hybrid"
        logger.info("[Base-Agent] No-memory mode: skipping all offline evolution and evaluator RAG updates")

    eval_skill_mode = eval_skill_mode_of(no_evaluator=no_evaluator,
                                         no_evolution=no_evolution,
                                         static_skills=static_skills)
    logger.info(f"[Calibration] Evaluator playbook mode={eval_skill_mode}"
                f" (live=episode-wise evolution / frozen=prewritten read-only / off=no loading or injection)")

    llm_cfg = load_config("llm.yaml")
    eff_qpm = int(llm_cfg.get("qpm", 0) or 0) if qpm is None else int(qpm)
    eff_workers = (int(llm_cfg.get("max_concurrency", 1) or 1)
                   if max_concurrency is None else int(max_concurrency))
    eff_workers = max(1, eff_workers)
    limiter = configure_rate_limiter(eff_qpm)
    if eff_workers > 1:
        logger.info(f"[Concurrency] Synthesis/evaluation workers = {eff_workers}"
                    f" (QPM={eff_qpm or 'unlimited'}; logs are grouped by task before printing)")
        if eff_qpm and eff_workers > eff_qpm:
            logger.warning(f"[Concurrency] Workers {eff_workers} exceed QPM limit {eff_qpm}; "
                           f"extra threads only wait at the rate limiter and do not improve throughput")
    explorer_llm = LLMClient(name="explorer")
    synth_llm = LLMClient(name="synthesizer")
    eval_llm = LLMClient(name="evaluator")
    skill_llm = LLMClient(name="skill")
    if data_dir:
        from ..execution.parquet_backend import ParquetExecutor
        data_dir = Path(data_dir).expanduser().resolve()
        offline_panel = offline_panel or (data_dir / "splits" / "offline_panel.parquet")
        executor = ParquetExecutor(data_dir, panel_path=offline_panel)
    else:
        executor = build_executor_from_config(enabled=eg_exec or explore_live)
    if executor is not None:
        if preflight(executor):
            logger.info(f"[Execution] Precheck passed (backend={getattr(executor, 'backend', 'easygraph')}, "
                        f"space={executor.space}); live execution validation enabled")
        else:
            logger.warning("[EasyGraph] Precheck failed; live execution remains enabled "
                           "(precheck does not block execution; per-step retries and reset handle "
                           "server warmup/outages, and execution resumes after recovery)")

    source = None
    min_hops = 0
    if explore_live:
        if executor is None:
            raise SystemExit("[Live exploration] EasyGraph connection required, but executor initialization failed "
                             "(check config/execution.yaml and network connectivity)")
        from ..execution.graph_source import (
            LiveGraphSource, RandomSampleSource, ParquetGraphSource,
            ParquetRandomSampleSource, ParquetTargetStream)
        from ..core.graph_clean import GraphCleaner
        from ..core.graph import entity_index
        if not data_dir:
            from ..data.target_stream import TargetStream
        caps = [int(x) for x in (per_hop_caps.split(",") if isinstance(per_hop_caps, str)
                                 else per_hop_caps) if str(x).strip()]
        exec_cfg = load_config("execution.yaml")
        live_cfg = exec_cfg.get("explore_live") or {}
        bl_cfg = live_cfg.get("blacklist") or {}
        res_dir = SYSTEM_ROOT / "data" / "graph_res"
        cleaner = GraphCleaner(
            vtype_blacklist_path=bl_cfg.get("vtype") or str(res_dir / "vtype_blacklist.txt"),
            vid_blacklist_path=bl_cfg.get("vid") or str(res_dir / "vid_blacklist.txt"),
            field_blacklist_path=bl_cfg.get("edge_field") or str(res_dir / "edge_field_blacklist.txt"))
        min_hops = int(live_cfg.get("min_hops", 3))
        probe_min_hops = int(live_cfg.get("probe_min_hops", min_hops))
        if data_dir:
            src_cls = ParquetRandomSampleSource if no_explorer else ParquetGraphSource
        else:
            src_cls = RandomSampleSource if no_explorer else LiveGraphSource
        source = src_cls(
            executor, cleaner, per_hop_caps=caps,
            edge_limit=int(live_cfg.get("edge_limit", 5000)),
            min_hops=min_hops, probe_min_hops=probe_min_hops,
            probe_max_frontier=int(live_cfg.get("probe_max_frontier", 8)),
            induced_edges=bool(live_cfg.get("induced_edges", True)),
            probe_quiet=bool(live_cfg.get("probe_quiet", True)),
            entity_names=list(entity_index().keys()),
            **({"seed": random_seed} if no_explorer else {}))
        if data_dir:
            stream = ParquetTargetStream(
                source, sample=target_sample, seed=random_seed,
                targets_path=targets_path)
        else:
            stream = TargetStream(source, sr_cfg=exec_cfg.get("starrocks") or {},
                                  account_type=account_type, sample=target_sample,
                                  seed=random_seed,
                                  cache_cfg=live_cfg.get("target_cache") or {})
        budgets = [1] + [int(c) for c in caps[:max_hops]]
        node_budgets = budgets
        logger.info(f"[Live exploration] Enabled: per-hop new-node caps {caps} -> cumulative visible budgets {budgets}, "
                    f"min_hops={min_hops}, {target_sample} targets "
                    f"(backend={'FraudFeat Parquet' if data_dir else 'EasyGraph'}, "
                    f"account_type={account_type})")
    else:
        stream = SubgraphStream(path=input_path)
    explorer = Explorer(explorer_llm, max_hops=max_hops,
                        budget_schedule=node_budgets, auto_topup=explore_topup,
                        strict_depth=strict_depth, global_review=global_review,
                        source=source, min_hops=min_hops,
                        emit_plan=no_synthesizer)
    syn_kw = synthesizer_kwargs_of(no_synthesizer=no_synthesizer,
                                   max_repair=max_repair,
                                   semantic_check=semantic_check)
    if no_synthesizer:
        logger.info("[Ablation] w/o Synthesizer: explorer produces hypotheses and plans in one call (no in-loop repair); "
                    "semantic checks, execution, consistency diagnosis, and zero-hit checks remain enabled "
                    "(same downstream gates as OURS)")
        if not semantic_check:
            logger.info("[Ablation] Semantic checks forced on (ablation removes repair, not gates)")
    synthesizer = Synthesizer(synth_llm, executor=executor, **syn_kw)
    eval_skill_memory = None
    _eval_book_path = None
    if eval_skill_mode == "frozen":
        _eval_book_path = Path(static_skills) / "eval_calibration.md"
    elif eval_skill_mode == "live":
        _sp = Path(skills_path)
        _md_dir = _sp if _sp.suffix.lower() != ".json" else _sp.parent / "skills"
        _eval_book_path = _md_dir / "eval_calibration.md"
    if _eval_book_path is not None:
        eval_skill_memory = EvalSkillMemory(
            skill_llm, store_path=_eval_book_path,
            persist=(eval_skill_mode == "live"),
            reset=(bool(reset_eval) if eval_skill_mode == "live" else False),
            read_only=(eval_skill_mode == "frozen"))
        if eval_skill_mode == "frozen" and not _eval_book_path.exists():
            logger.info(f"[Calibration] Static-Agent frozen assets lack {_eval_book_path.name}"
                        f" (legacy asset compatibility); no empty block injected")
        else:
            logger.info(f"[Calibration] Loaded evaluator playbook (mode={eval_skill_mode}, "
                        f"{eval_skill_memory.card_count()} entries) -> {_eval_book_path}"
                        + (" (frozen read-only: no revision or persistence)"
                           if eval_skill_mode == "frozen" else ""))
    evaluator = Evaluator(eval_llm, retriever_mode=retriever_mode,
                          eval_skill_memory=eval_skill_memory)
    keep_eval = (coldstart.protected_eval_doc_names()
                 if reset_eval else set()) or None
    eval_memory = EvaluatorMemory(eval_llm, store_path=eval_memory_path,
                                  persist=persist_eval, reset=reset_eval,
                                  keep_doc_names=keep_eval, read_only=_frozen_read)
    if eval_seeds_path:
        if len(eval_memory) == 0:
            seeds = _load_eval_seeds(eval_seeds_path)
            eval_memory.bootstrap(seeds)
            logger.info(f"[Evaluator] Cold start: injected {len(seeds)} seeds from {eval_seeds_path}"
                        f" (uploaded to RAG via warm_items during the offline phase)")
        else:
            logger.info(f"[Evaluator] Memory already has {len(eval_memory)} entries; skipping cold-start seeds")
    keep_skills = (coldstart.protected_skill_slugs()
                   if reset_skills else set()) or None
    skill_memory = SkillMemory(skill_llm, store_path=skills_path,
                               persist=persist_skills, reset=reset_skills,
                               keep_slugs=keep_skills, read_only=_frozen_read)
    skill_memory.register_paradigms()
    diversity_tracker = ParadigmDiversityTracker(
        path=(out_dir / "memory" / "diversity_state.json") if run_id else None,
        persist=persist_skills)
    if reset_skills or reset_eval or reset_synth:
        diversity_tracker.reset()
    synthesis_memory = SynthesisSkillMemory(
        skill_llm, store_path=synth_skills_path,
        persist=persist_synth, reset=reset_synth)
    if persist_skills:
        _cc = getattr(skill_memory, "card_count", None)
        if callable(_cc) and _cc() == 0:
            logger.info("[Skills] Paradigm, navigation, and hypothesis playbooks are empty (first run or reset); "
                        "injection points use empty-playbook notices, and offline learning starts this episode")
    if enrich_skills:
        try:
            _n1 = skill_memory.enrich_cards()
            _n2 = synthesis_memory.enrich_cards()
            logger.info(f"[Skills] Enriched {_n1} exploration entries and {_n2} synthesis entries")
        except Exception as e:
            logger.warning(f"[Skills] Enrichment failed ({e}); skipping without affecting the main flow")

    coldstart.run_for(account_type=account_type, eval_memory=eval_memory,
                      skill_memory=skill_memory, llm=skill_llm,
                      cli_enabled=cold_start, cli_reset=cold_start_reset,
                      persist_ok=(persist_eval and persist_skills))

    try:
        n_warm = 0 if no_evaluator else evaluator.warm_memory(eval_memory.memory)
        if n_warm:
            logger.info(f"[Retrieval] Startup warmup: uploaded/cached {n_warm} evaluator memories for first-episode retrieval")
    except Exception as e:
        logger.warning(f"[Retrieval] Startup warmup failed ({e}); fallback on demand during online evaluation")
    gate = Gate(GateConfig())
    threshold_policy = ThresholdPolicy(mode=threshold_mode)
    logger.info(f"[Thresholds] Policy mode={threshold_policy.mode}"
                f" (fixed=interpretable fixed thresholds / adaptive=batch quantiles; fallback ensures positive and negative samples)")
    pool: list[Candidate] = []
    queue: list[Candidate] = []
    history: list[dict] = []
    recent_guidance: str = ""
    process_negatives: list[Candidate] = []
    total_subgraphs = 0
    if out_dir is None:
        out_dir = ensure_dir(_WORKSPACE / "runs" / "default")
    start_episode = 0
    if episode_resume and run_id and out_dir:
        _ws = _restore_episode_state(out_dir)
        if _ws is not None:
            start_episode, pool, history, total_subgraphs = _ws
            queue = _restore_candidate_checkpoint(out_dir / "queue_checkpoint.json")
            process_negatives = _restore_candidate_checkpoint(
                out_dir / "synthesis_failures_checkpoint.json")
            process_negatives = [c for c in process_negatives
                                 if not getattr(c, "_n_consumed", False)]
            logger.info(f"[resume] Restored {len(queue)} queued candidates and "
                        f"{len(process_negatives)} pending synthesis failures")
        else:
            logger.info("[Episode resume] No recoverable progress; starting at episode 0")
    eff_episodes = max_episodes if max_episodes is not None else num_episodes
    if valid_target:
        logger.info(f"[Stopping rule] Stop after {valid_target} valid features"
                    f" (maximum {max_episodes} episodes; record cum_valid/cum_subgraphs each episode)")
    for w in range(start_episode, eff_episodes):
        _disk_guard(out_dir)
        if verbose:
            logger.info(f"[Online] Episode {w}: processing {episode_size} subgraphs (explore -> synthesize -> evaluate -> queue)")
        produced_this_episode = 0

        handled = 0
        skipped_targets = 0
        episode_cands: list[Candidate] = []
        pending_steer: str = ""
        if recent_guidance and verbose:
            logger.info(f"[Online] Episode {w}: injecting previous-episode exploration guidance "
                        f"({len(recent_guidance)} characters: ground-truth evidence x coverage) "
                        f"to steer Explorer")
        while handled < episode_size:
            sg = stream.next()
            if sg is None:
                break
            cands, _steer = process_subgraph(
                sg, explorer=explorer, synthesizer=synthesizer,
                evaluator=evaluator, eval_memory=eval_memory,
                skill_memory=skill_memory, pool=pool, queue=queue,
                synthesis_memory=synthesis_memory,
                diversity_tracker=(None if no_diversity_guide else diversity_tracker),
                episode_index=w,
                max_workers=eff_workers,
                guidance_block=recent_guidance,
                online_feedback=pending_steer,
                no_evaluator=no_evaluator, no_explorer=no_explorer,
                no_synthesizer=no_synthesizer)
            if _steer:
                pending_steer = _steer
            if not cands:
                skipped_targets += 1
                logger.info(f"[Online] Subgraph {getattr(sg, 'center', '?')}: shallow or no output"
                            f" (replacement {skipped_targets}); fetching next target")
                continue
            handled += 1
            episode_cands += cands
            produced_this_episode += len([c for c in cands if not getattr(c, "_negative", False)])
            process_negatives += [c for c in cands
                                  if getattr(c, "_negative", False)
                                  or getattr(c, "_err_history", None)]
            total_subgraphs += 1
            if max_subgraphs and total_subgraphs >= max_subgraphs:
                break
        if skipped_targets:
            logger.info(f"[Online] Episode {w}: replaced {skipped_targets} shallow/no-output targets, "
                        f"valid subgraphs {handled}/{episode_size}")
        if total_subgraphs == 0:
            logger.warning("No subgraphs to process; stopping")
            break
        force_budget = (w % 1 == 0)
        should, reason = gate.should_ground(len(queue), force_budget)
        if should and queue:
            if verbose:
                logger.info(_OFFLINE_DIVIDER)
                logger.info(f"[Offline] Episode {w} grounding triggered: {reason} ({len(queue)} queued candidates awaiting true IV)")
                logger.info("[Offline] Steps: 1) select top-K -> 2) batch ComputeTrueIV -> 3) update evaluator memory -> 4) update feature pool -> 5) evolve skills")
                logger.info(_OFFLINE_DIVIDER)
            info = run_offline(
                queue=queue, pool=pool, eval_memory=eval_memory,
                skill_memory=skill_memory, gate=gate, m_episodes=m_episodes,
                episode_index=w, force_budget=force_budget,
                threshold_policy=threshold_policy,
                process_negatives=process_negatives,
                synthesis_memory=synthesis_memory,
                diversity_tracker=(None if no_diversity_guide else diversity_tracker),
                no_evaluator=no_evaluator,
                no_evolution=no_evolution,
                no_diversity_guide=no_diversity_guide,
                eval_skill_memory=eval_skill_memory,
                executor=executor)
            if info.get("skill_evolved"):
                for _c in process_negatives:
                    _c._n_consumed = True
            if no_evolution and len(eval_memory):
                _clr = getattr(eval_memory, "clear", None)
                if callable(_clr):
                    _clr()
                    logger.info("[Base-Agent] Memory cleared at the episode boundary to prevent carryover")
            if not no_evaluator:
                evaluator.warm_memory(eval_memory.memory)
            recent_guidance = info.get("guidance") or ""
            history.append({
                "episode": w, "reason": reason,
                "grounded": info["grounded"],
                "entered_pool": info["entered_pool"],
                "pool_size": len(pool),
                "cum_valid": len(pool),
                "cum_subgraphs": total_subgraphs,
                "new_skills": info["new_skills"],
                "eval_mem_size": len(eval_memory),
                "pos_samples": info.get("pos_samples", 0),
                "neg_samples": info.get("neg_samples", 0),
                "queue_remaining": info.get("queue_remaining", len(queue)),
                "episode_stats": info.get("episode_stats"),
            })
            if verbose:
                _log_episode(history[-1])
        else:
            if verbose:
                logger.info(f"[episode {w}] Grounding not triggered (queue size {len(queue)})")
        _checkpoint(out_dir, pool, history, skill_memory, eval_memory,
                    queue=queue, process_negatives=process_negatives)
        _write_candidates(out_dir, episode_cands, pool, w)
        _write_case_materials(out_dir, episode_cands, pool, w)
        if max_subgraphs and total_subgraphs >= max_subgraphs:
            break
        if valid_target and len(pool) >= valid_target:
            logger.info(f"[stop rule] {len(pool)} valid features >= {valid_target}; target reached "
                        f"({total_subgraphs} subgraphs processed, episode {w + 1})")
            break
    eval_memory.save()
    skill_memory.save()
    synthesis_memory.save()
    _stream_close = getattr(stream, "close", None)
    if callable(_stream_close):
        _stream_close()
    if executor is not None:
        executor.close()
    summary = {
        "run_id": run_id,
        "out_dir": str(out_dir),
        "episodes": num_episodes,
        "subgraphs_processed": total_subgraphs,
        "final_pool_size": len(pool),
        "final_pool": [c.to_dict() for c in pool],
        "features_export": _features_payload(pool, history),
        "skills": [s.to_dict() for s in skill_memory.skills],
        "skills_store": str(skill_memory.store_path) if skill_memory.persist else None,
        "synth_skills": [s.to_dict() for s in synthesis_memory.skills],
        "synth_skills_store": (str(synthesis_memory.store_path)
                               if synthesis_memory.persist else None),
        "eval_memory_store": str(eval_memory.store_path) if eval_memory.persist else None,
        "eval_skill_store": (
            str(eval_skill_memory.store_path)
            if (eval_skill_memory is not None and eval_skill_memory.persist) else None),
        "eval_skill_mode": eval_skill_mode,
        "explorer_budgets": explorer.hop_budgets(),
        "eval_mem_size": len(eval_memory),
        "llm_calls": explorer_llm.calls + synth_llm.calls + eval_llm.calls + skill_llm.calls,
        "concurrency": eff_workers,
        "rate_limit": limiter.stats(),
        "history": history,
        "threshold_mode": threshold_policy.mode,
        "retriever_mode": retriever_mode,
        "semantic_check": syn_kw.get("semantic_check", semantic_check),
        "max_repair": syn_kw.get("max_repair", max_repair),
        "eg_exec": executor is not None,
        "explore_live": explore_live,
        "data_dir": str(data_dir) if data_dir else None,
        "offline_panel": (str(executor.panel_path)
                          if getattr(executor, "panel_path", None) else None),
        "targets_path": (str(stream.targets_path)
                         if getattr(stream, "targets_path", None) else None),
        "random_seed": random_seed,
        "evaluation": {
            "backend": "fraudfeat" if data_dir else "easygraph",
            "amount_unit": "fen",
            "id_list_limit": 300,
        },
    }
    return summary


def _log_episode(rec: dict) -> None:
    logger.info(
        f"  admitted={rec['entered_pool']} | pool_size={rec['pool_size']} | "
        f"new_skills={rec['new_skills']} | "
        f"memory_size={rec['eval_mem_size']}")


def _features_payload(pool: list[Candidate], history: list[dict]) -> dict:
    feats = [feature_export_entry(c) for c in pool]
    for i, d in enumerate(feats):
        d["pool_index"] = i
        d["admission_index"] = i
    return {"episode_stats": [h["episode_stats"] for h in history
                             if h.get("episode_stats")],
            "features": feats}


def _explanation_fallback(c: Candidate) -> str:
    parts = []
    rp = (getattr(c, "fraud_pattern", "") or "").strip()
    if rp:
        parts.append(rp.rstrip(".") + ".")
    if c.rationale:
        parts.append(str(c.rationale).strip().rstrip(".") + ".")
    return " ".join(parts)


def feature_export_entry(c: Candidate) -> dict:
    return {
        "feature_name": c.feature_name,
        "feature_name_cn": getattr(c, "feature_name_cn", "") or "",
        "episode": getattr(c, "episode", None),
        "true_tier": (IVTier.from_value(c.true_iv).value
                      if c.true_iv is not None else None),
        "business_explanation": (getattr(c, "business_explanation", "") or "")
                                 or _explanation_fallback(c),
        "gremlin_plan": getattr(c, "gremlin_plan", None) or {},
        "query": c.dsl_program,
        "value_type": c.value_type,
        "predicted_tier": c.predicted_tier.value if c.predicted_tier else None,
        "predicted_conf": c.predicted_conf,
        "predicted_iv": getattr(c, "predicted_iv", None),
        "true_iv": c.true_iv,
        "rationale": c.rationale,
        "fraud_pattern": getattr(c, "fraud_pattern", "") or "",
        "evidence_chain": getattr(c, "evidence_chain", None),
        "expected_value_type": getattr(c, "expected_value_type", "") or "",
        "logic": getattr(c, "logic", "") or "",
        "consistency_report": getattr(c, "consistency_report", "") or "",
        "consistency_detail": getattr(c, "consistency_detail", None) or {},
        "feature_value": getattr(c, "feature_value", None),
        "verify": {
            "center_vid": getattr(c, "center_vid", None),
            "evidence": getattr(c, "evidence", None),
            "how": "Replace {{SEED}} in gremlin_plan with \"<center_vid>|entity\", submit each step in order "
                   "(replacing {{STEP:sX}} with the upstream token list), and reduce final in Python; the result "
                   "is the feature value for that account. Check the structure against evidence_chain.",
        },
    }


_CHECKPOINT_TRANSIENT_FIELDS = (
    "_err_history", "_failed_plan", "_synth_vocab", "_synth_trace",
    "_skills_used", "_explorer_skills", "_navigator_history",
    "_negative", "_n_consumed", "_synth_fallback", "_explorer_fault",
)


def _candidate_checkpoint_payload(c: Candidate) -> dict:
    data = c.to_dict()
    data["trace"] = [{"action": t.action, "obs": t.obs, "thought": t.thought,
                      "evidence": t.evidence} for t in (c.trace or [])]
    data["validation_report"] = c.validation_report
    data["predicted_reason"] = getattr(c, "predicted_reason", "") or ""
    data["transient"] = {
        key: getattr(c, key) for key in _CHECKPOINT_TRANSIENT_FIELDS
        if hasattr(c, key)
    }
    return data


def _restore_candidate_checkpoint(path: Path) -> list[Candidate]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("candidate checkpoint must contain a list")
        restored = []
        for item in data:
            if not isinstance(item, dict):
                continue
            c = Candidate.from_dict(item)
            c.validation_report = str(item.get("validation_report") or "")
            c.predicted_reason = str(item.get("predicted_reason") or "")
            c.trace = [TraceStep(
                action=str(t.get("action") or ""), obs=str(t.get("obs") or ""),
                thought=str(t.get("thought") or ""),
                evidence=list(t.get("evidence") or []))
                for t in (item.get("trace") or []) if isinstance(t, dict)]
            transient = item.get("transient") or {}
            if isinstance(transient, dict):
                for key in _CHECKPOINT_TRANSIENT_FIELDS:
                    if key in transient:
                        setattr(c, key, transient[key])
            restored.append(c)
        return restored
    except Exception as e:
        logger.warning(f"[resume] Candidate checkpoint {path} read failed: {e}")
        return []


def _checkpoint(out_dir, pool, history, skill_memory, eval_memory,
                queue=None, process_negatives=None) -> None:
    from ..llm.client import aggregate_usage
    from ..execution.eg_client import get_query_count
    (out_dir / "features.json").write_text(
        json.dumps(_features_payload(pool, history),
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    prog = {
        "episodes_done": len(history),
        "pool_size": len(pool),
        "skills": [s.to_dict() for s in skill_memory.skills],
        "eval_mem_size": len(eval_memory),
        "history": history,
        "usage": {**aggregate_usage(), "graph_queries": get_query_count()},
    }
    (out_dir / "progress.json").write_text(
        json.dumps(prog, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out_dir / "pool_checkpoint.json").write_text(
        json.dumps([c.to_dict() for c in pool],
                   ensure_ascii=False, default=str), encoding="utf-8")
    if queue is not None:
        (out_dir / "queue_checkpoint.json").write_text(
            json.dumps([_candidate_checkpoint_payload(c) for c in queue],
                       ensure_ascii=False, default=str), encoding="utf-8")
    if process_negatives is not None:
        pending = [c for c in process_negatives
                   if not getattr(c, "_n_consumed", False)]
        (out_dir / "synthesis_failures_checkpoint.json").write_text(
            json.dumps([_candidate_checkpoint_payload(c) for c in pending],
                       ensure_ascii=False, default=str), encoding="utf-8")
    try:
        eval_memory.save()
    except Exception as e:
        logger.warning(f"[checkpoint] Evaluator memory save failed (main run continues): {e}")
    logger.info(f"[checkpoint] Episode saved: {len(pool)} pooled features -> {out_dir / 'features.json'}")


def _restore_episode_state(out_dir: Path) -> tuple[int, list, list, int] | None:
    prog_path = Path(out_dir) / "progress.json"
    if not prog_path.exists():
        return None
    try:
        prog = json.loads(prog_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    episodes_done = int(prog.get("episodes_done", prog.get("windows_done")) or 0)
    history = list(prog.get("history") or [])
    for record in history:
        if "episode" not in record and "window" in record:
            record["episode"] = record.pop("window")
        if "episode_stats" not in record and "window_stats" in record:
            record["episode_stats"] = record.pop("window_stats")
    if episodes_done <= 0:
        return None
    pool = _restore_pool(Path(out_dir))
    total_subgraphs = int(history[-1].get("cum_subgraphs") or 0) if history else 0
    logger.info(f"[resume] Continuing from episode {episodes_done}: {len(pool)} pooled features, "
                f"{total_subgraphs} subgraphs already processed")
    return episodes_done, pool, history, total_subgraphs


def _restore_pool(out_dir: Path) -> list:
    ckpt_path = Path(out_dir) / "pool_checkpoint.json"
    if ckpt_path.exists():
        try:
            data = json.loads(ckpt_path.read_text(encoding="utf-8"))
            pool = [Candidate.from_dict(d) for d in data]
            logger.info(f"[resume] Pool restored from pool_checkpoint.json ({len(pool)} features)")
            return pool
        except Exception as e:
            logger.warning(f"[resume] pool_checkpoint.json read failed ({e}); trying features.json")
    feat_path = Path(out_dir) / "features.json"
    if feat_path.exists():
        try:
            data = json.loads(feat_path.read_text(encoding="utf-8"))
            feats = data.get("features", []) if isinstance(data, dict) else data
            pool = [_candidate_from_export(d) for d in feats]
            logger.info(f"[resume] Pool restored from features.json ({len(pool)} features; "
                        f"missing export fields filled with defaults)")
            return pool
        except Exception as e:
            logger.warning(f"[resume] features.json restore failed ({e})")
    logger.warning("[resume] No recoverable pool; starting with an empty pool")
    return []


def _candidate_from_export(d: dict) -> "Candidate":
    c = Candidate(
        feature_name=d["feature_name"],
        dsl_program=d.get("query") or d.get("gremlin_plan") or {},
        value_type=d.get("value_type", "count"),
        rationale=d.get("rationale", ""),
        predicted_tier=IVTier(d.get("predicted_tier", "low")),
        predicted_conf=float(d.get("predicted_conf", 0.0)),
    )
    c.episode = d.get("episode", d.get("window"))
    c.feature_name_cn = d.get("feature_name_cn", "")
    c.business_explanation = d.get("business_explanation", "")
    c.predicted_iv = d.get("predicted_iv")
    c.true_iv = d.get("true_iv")
    c.grounded = True
    c.synth_ok = True
    c.fraud_pattern = d.get("fraud_pattern", d.get("risk_pattern", ""))
    c.evidence_chain = d.get("evidence_chain") or []
    c.expected_value_type = d.get("expected_value_type", "")
    c.logic = d.get("logic", "")
    c.consistency_report = d.get("consistency_report", "")
    c.gremlin_plan = d.get("gremlin_plan") or {}
    c.feature_value = d.get("feature_value")
    c.consistency_detail = d.get("consistency_detail") or {}
    verify = d.get("verify") or {}
    c.center_vid = verify.get("center_vid", "")
    c.evidence = verify.get("evidence") or []
    return c


def _checkpoint_base(out_dir, pool, history) -> None:
    from ..llm.client import aggregate_usage
    from ..execution.eg_client import get_query_count
    (Path(out_dir) / "features.json").write_text(
        json.dumps(_features_payload(pool, history),
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (Path(out_dir) / "progress.json").write_text(
        json.dumps({"episodes_done": len(history), "pool_size": len(pool),
                    "history": history,
                    "usage": {**aggregate_usage(),
                              "graph_queries": get_query_count()}},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _sig_str(c: Candidate) -> str:
    try:
        rels, terminal, evt, (attrs, stat) = structure_signature(c)
        return ">".join(rels) + f"|{terminal}|{evt}|{','.join(attrs)}|{stat}"
    except Exception:
        return ""


def _repairs_of(c: Candidate) -> int:
    return sum(1 for t in (getattr(c, "trace", None) or [])
               if str(getattr(t, "action", "")).startswith("ReAct"))


def _outcome_of(c: Candidate, pooled: bool) -> str:
    if pooled:
        return "pooled"
    if not getattr(c, "synth_ok", True):
        return "synth_failed"
    if getattr(c, "_stratum", None) and c.true_iv is None:
        return "frozen"
    return "truncated"


def _candidate_line(c: Candidate, outcome: str) -> dict:
    err_hist = [str(x) for x in (getattr(c, "_err_history", None) or [])]
    last_err = (err_hist[-1] if err_hist else (getattr(c, "exec_error", "") or ""))
    return {
        "feature_name": c.feature_name,
        "feature_name_cn": getattr(c, "feature_name_cn", "") or "",
        "logic": getattr(c, "logic", "") or "",
        "fraud_pattern": getattr(c, "fraud_pattern", "") or "",
        "evidence_chain": getattr(c, "evidence_chain", None),
        "expected_value_type": getattr(c, "expected_value_type", "") or "",
        "episode": getattr(c, "episode", None),
        "outcome": outcome,
        "error_signature": (error_signature(last_err) if last_err else ""),
        "stratum": getattr(c, "_stratum", None),
        "death_stage": getattr(c, "_death_stage", None),
        "repairs": _repairs_of(c),
        "exec_attempts": getattr(c, "exec_attempts", 0),
        "exec_error": (getattr(c, "exec_error", "") or "")[:120],
        "skills_used": list(getattr(c, "_skills_used", []) or []),
        "sig": _sig_str(c),
        "predicted_iv": getattr(c, "predicted_iv", None),
        "true_iv": c.true_iv,
    }


def _write_candidates(out_dir: Path, episode_cands: list[Candidate],
                      pool: list[Candidate], episode: int) -> None:
    if not episode_cands:
        return
    pooled = {c.feature_name for c in pool}
    path = Path(out_dir) / "candidates.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        for c in episode_cands:
            outcome = _outcome_of(c, c.feature_name in pooled)
            if outcome == "pooled":
                rec = feature_export_entry(c)
                rec.update(_candidate_line(c, outcome))
                rec.pop("verify", None)
            else:
                rec = _candidate_line(c, outcome)
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


_CASE_CAP_PER_EPISODE = 4


def _case_worthy(c: Candidate) -> bool:
    if _repairs_of(c) > 0:
        return True
    if getattr(c, "_explorer_fault", False):
        return True
    return c.true_iv is not None and c.true_iv >= 0.5


def _write_case_materials(out_dir: Path, episode_cands: list[Candidate],
                          pool: list[Candidate], episode: int) -> None:
    picked = [c for c in episode_cands if _case_worthy(c)][: _CASE_CAP_PER_EPISODE]
    if not picked:
        return
    pooled = {c.feature_name for c in pool}
    case_dir = ensure_dir(Path(out_dir) / "cases")
    for c in picked:
        rec = {
            "episode": episode,
            "feature_name": c.feature_name,
            "outcome": _outcome_of(c, c.feature_name in pooled),
            "entry": feature_export_entry(c),
            "trace": [vars(t) if hasattr(t, "__dict__") else dict(t)
                      for t in (getattr(c, "trace", None) or [])],
            "synth_trace": getattr(c, "_synth_trace", None) or [],
            "err_history": [str(x) for x in (getattr(c, "_err_history", None) or [])],
            "failed_plan": getattr(c, "_failed_plan", None),
            "consistency_detail": getattr(c, "consistency_detail", None) or {},
        }
        name = re.sub(r'[\\/:*?"<>|\r\n\t\s]+', "_", c.feature_name)[:60] or "feature"
        (case_dir / f"w{episode}_{name}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8")
