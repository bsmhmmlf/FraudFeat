from __future__ import annotations

import importlib
import os

from .utils import get_logger, load_config


logger = get_logger("selfcheck")
_CHECKS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("system.execution",
     ("execute_plan", "render_execution_trace", "normalize_plan", "sanitize_plan",
      "validate_plan", "sanitize_vid", "vid_to_token",
      "build_executor_from_config", "preflight", "is_transient_exec_error",
      "is_slow_query_error", "SERVER_OUTAGE_ERR_PREFIX", "SLOW_QUERY_ERR_PREFIX"),
     "Execution exports (eg_plan/eg_client)"),
    ("system.execution.graph_source",
     ("LiveGraphSource", "ParquetGraphSource", "ParquetTargetStream"),
     "Live graph sources"),
    ("system.execution.parquet_backend", ("ParquetExecutor",), "Public Parquet executor"),
    ("system.core.schema", ("Candidate", "IVTier"), "Candidates and IV tiers"),
    ("system.core.graph", ("Subgraph",), "Subgraph structure"),
    ("system.evolution.reward", ("evaluate_batch",), "Batch measurement entry point"),
    ("system.evolution.true_iv", ("compute_true_iv", "simulate_true_iv"), "Measured IV routing"),
    ("system.evolution.family", ("candidate_similarity", "structure_signature"), "Structural signatures and redundancy"),
    ("system.agents.explorer", ("Explorer",), "Explorer"),
    ("system.agents.synthesizer", ("Synthesizer",), "Synthesizer"),
    ("system.agents.evaluator", ("Evaluator",), "Evaluator"),
    ("system.agents.consistency", ("diagnose_consistency",), "Consistency diagnosis"),
    ("system.orchestrator.online_phase", ("process_subgraph",), "Online phase"),
    ("system.orchestrator.offline_phase", ("run_offline", "_synthesis_episode"), "Offline phase"),
    ("system.orchestrator.loop",
     ("run", "_write_candidates", "_write_case_materials", "_checkpoint_base"),
     "Main loop and shared baseline persistence interfaces"),
    ("system.evolution.skill_memory", ("SkillMemory",), "Explorer playbooks"),
    ("system.evolution.eval_skill_memory", ("EvalSkillMemory",), "Evaluator calibration playbook"),
    ("system.evolution.synthesis_skill_memory", ("SynthesisSkillMemory",), "Synthesis playbook"),
    ("system.evolution.evaluator_eval", ("EvaluatorMemory",), "Evaluator memory"),
    ("system.evolution.failure_taxonomy", ("error_signature",), "Failure taxonomy"),
    ("system.evolution.coldstart", ("run_for",), "Cold start"),
    ("system.evolution.thresholds", ("ThresholdPolicy",), "Threshold policy"),
    ("system.utils", ("get_logger", "load_config", "ensure_dir"), "Utilities"),
    ("system.llm.client", ("LLMClient",), "LLM client"),
    ("system.parallel", ("run_parallel",), "Parallel execution"),
)
_CHECKS_REAL: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("system.evolution.real_iv",
     ("compute_real_iv", "real_mode_enabled"),
     "Panel IV measurement (iv.yaml mode=real)"),
)
_SKIP_ENV = "SYSTEM_SKIP_SELFCHECK"


def _real_mode() -> bool:
    try:
        return str(load_config("iv.yaml").get("mode") or "").strip().lower() == "real"
    except Exception:
        return True


def check(need_real_iv: bool | None = None) -> list[str]:
    if need_real_iv is None:
        need_real_iv = _real_mode()
    items = _CHECKS + (_CHECKS_REAL if need_real_iv else ())
    missing: list[str] = []
    for mod_name, symbols, purpose in items:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            missing.append(f"{mod_name} import failed ({type(e).__name__}: {e}) | {purpose}")
            continue
        for sym in symbols:
            if not hasattr(mod, sym):
                missing.append(f"{mod_name} lacks symbol {sym} | {purpose}")
    return missing


def verify(stage: str, *, need_real_iv: bool | None = None) -> list[str]:
    missing = check(need_real_iv=need_real_iv)
    if not missing:
        logger.info(f"[startup check] Dependencies complete ({stage}: modules import and required symbols exist)")
        return missing
    detail = "\n".join(f"    - {m}" for m in missing)
    msg = (f"[startup check] {len(missing)} missing dependencies ({stage}):\n{detail}\n"
           f"    A partial deployment is a likely cause, such as updated modules without "
           f"matching package __init__ exports.\n"
           f"    Online queries may succeed while ComputeTrueIV measurements remain frozen, "
           f"wasting an episode without an obvious run error.\n"
           f"    Synchronize the complete system/ and experiments/ directories, then restart. "
           f"Set {_SKIP_ENV}=1 only if you intentionally accept warning-only startup.")
    if os.environ.get(_SKIP_ENV, "").strip() in ("1", "true", "yes"):
        logger.warning(msg + f"\n    ({_SKIP_ENV} is set: warning only; continuing startup)")
        return missing
    raise RuntimeError(msg)
