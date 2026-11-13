from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from .orchestrator.loop import run
from .paths import RUNS_DIR
from .utils import ensure_dir, get_logger

logger = get_logger("main")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="EvoFeat: automating fraud-graph feature mining with self-evolving agents")
    parser.add_argument("--episodes", type=int, default=3, help="Number of online episodes")
    parser.add_argument("--episode-size", type=int, default=2, help="Subgraphs per episode")
    parser.add_argument("--max-subgraphs", type=int, default=None, help="Maximum subgraphs to process")
    parser.add_argument("--m-episodes", type=int, default=1,
                        help="Skill evolution interval in episodes (default 1; larger values for ablation)")
    parser.add_argument("--max-repair", type=int, default=3, help="Maximum Synthesizer ReAct repair rounds")
    parser.add_argument("--max-hops", type=int, default=None,
                        help="Maximum Explorer hops; the LLM may stop early. "
                             "Default: 3 in static mode, length of --per-hop-caps in live mode")
    parser.add_argument("--max-refine", type=int, default=3, help="Legacy alias for --max-hops")
    parser.add_argument("--node-budgets", type=str, default="1,50,250,500,1000,2000",
                        help="Cumulative retained-node budget |U| per hop (hop 0 = target only)")
    parser.add_argument("--explore-topup", action="store_true",
                        help="Automatically fill unused node budget by bridge/hub ranking (off by default)")
    parser.add_argument("--strict-depth", action="store_true",
                        help="Expand one adjacent layer per hop without deeper prefetching")
    parser.add_argument("--no-global-review", action="store_true",
                        help="Disable global review that removes noise and recovers or adds missed nodes "
                             "before hypothesis generation (ablation)")
    parser.add_argument("--skills-path", type=str, default=None,
                        help="Playbook directory holding the five skill Markdown files "
                             "(default data/memory/skills/); legacy skills.json is "
                             "migrated to a derived directory and removed")
    parser.add_argument("--reset-skills", action="store_true",
                        help="Delete learned skill files and start learning from scratch")
    parser.add_argument("--no-skill-memory", action="store_true",
                        help="Do not read or write skill files in this run")
    parser.add_argument("--eval-memory-path", type=str, default=None,
                        help="Evaluator memory path (default data/memory/eval_memory.json)")
    parser.add_argument("--reset-eval-memory", action="store_true",
                        help="Delete evaluator memory and restart from seeds")
    parser.add_argument("--no-eval-memory", action="store_true",
                        help="Do not read or write evaluator memory in this run")
    parser.add_argument("--synth-skills-path", type=str, default=None,
                        help="Synthesis playbook path (default data/memory/skills/synthesis_playbook.md); "
                             "legacy synthesis_skills.json is migrated to Markdown and removed")
    parser.add_argument("--reset-synth-skills", action="store_true",
                        help="Delete learned synthesis skills and restart from scratch")
    parser.add_argument("--no-synth-skill-memory", action="store_true",
                        help="Do not read or write synthesis skills in this run")
    parser.add_argument("--eval-seeds-path", type=str, default=None,
                        help="Optional evaluator seed JSON, injected only into an empty memory and "
                             "then uploaded to RAG via warm_items. Disabled by default. "
                             "Template: config/eval_seeds.example.json")
    parser.add_argument("--cold-start", action="store_true",
                        help="Clean sources from config/coldstart.yaml and inject evaluator memory "
                             "and exploration skills at startup; markers make this idempotent. "
                             "Only sources matching --account-type are processed")
    parser.add_argument("--cold-start-reset", action="store_true",
                        help="Remove marker-tracked cold-start entries for --account-type, preserving "
                             "cleaned audit artifacts; combine with --cold-start to rebuild")
    parser.add_argument("--enrich-skills", action="store_true",
                        help="Fill missing skill names and scenarios once (at most one LLM call "
                             "for exploration and one for synthesis); idempotent, "
                             "existing values are preserved")
    parser.add_argument("--reset-memory", action="store_true",
                        help="Reset exploration, synthesis, and evaluator memories")
    parser.add_argument("--no-persist-memory", action="store_true",
                        help="Do not read or write any memory files in this run")
    parser.add_argument("--qpm", type=int, default=None,
                        help="Shared LLM calls-per-minute limit; "
                             "default from config/llm.yaml, 0 means unlimited")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Parallel synthesis/evaluation tasks (default config/llm.yaml "
                             "max_concurrency). 1 means serial; --qpm limits throughput, "
                             "so exceeding qpm does not improve speed")
    parser.add_argument("--threshold-mode", type=str, default="fixed",
                        choices=["fixed", "adaptive"],
                        help="Explorer skill labels only (all measured candidates enter the pool): "
                             "fixed uses IV 0.2 as the positive/negative boundary; "
                             "adaptive uses batch IV quantiles (top30%% positive/bottom30%% negative). "
                             "Both ensure positive and negative samples via head/tail fallback")
    parser.add_argument("--retriever-mode", type=str, default="hybrid",
                        choices=["hybrid", "embedding", "token"],
                        help="Evaluator memory retrieval: hybrid (embedding with token fallback), "
                             "embedding, or token")
    parser.add_argument("--no-semantic-check", action="store_true",
                        help="Disable chain continuity and endpoint-type checks (ablation); "
                             "retain executability and name/implementation consistency checks")
    parser.add_argument("--eg-exec", action="store_true",
                        help="Enable live EasyGraph execution and hy3 consistency diagnosis; "
                             "requires Linux and easygraph-v3-client-python. "
                             "Settings: config/execution.yaml; environment overrides include "
                             "SYSTEM_EG_ENABLED=1. A failed precheck falls back to offline mode")
    parser.add_argument("--explore-live", action="store_true",
                        help="Sample target vids from StarRocks and explore EasyGraph live with "
                             "cleaning, per-hop caps, and grouped overflow selection. "
                             "Features require at least 3 hops; enables --eg-exec. "
                             "Settings: config/execution.yaml starrocks/explore_live sections")
    parser.add_argument("--data", type=str, default=None, metavar="DIR",
                        help="Released FraudFeat data directory (schema.json, nodes/, edges/, "
                             "labels.parquet). Enables live exploration and execution directly "
                             "on Parquet without EasyGraph or StarRocks")
    parser.add_argument("--offline-panel", type=str, default=None, metavar="FILE",
                        help="Fixed offline IV panel (default <data>/splits/offline_panel.parquet; "
                             "shipped with the dataset)")
    parser.add_argument("--targets", type=str, default=None, metavar="FILE",
                        help="Official discovery-target JSON "
                             "(default <data>/splits/target_streams_s<seed>.json)")
    parser.add_argument("--account-type", type=str, default="individual",
                        choices=["individual"],
                        help="Public label population: wechat_pay_account")
    parser.add_argument("--target-sample", type=int, default=100,
                        help="Number of target vids sampled in live mode (default 100)")
    parser.add_argument("--random-seed", type=int, default=None,
                        help="Discovery/sampling seed (default: 42 with --data; "
                             "20260825 otherwise)")
    parser.add_argument("--per-hop-caps", type=str, default="50,250,500,1000,2000",
                        help="Comma-separated new-node caps per hop in live mode; length sets the "
                             "default hop limit (default 50,250,500,1000,2000)")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging")
    parser.add_argument("--input", type=str, default=None,
                        help="Input JSON of target-centered subgraphs; "
                             "default example.json at the workspace root")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Isolate outputs under runs/<run_id>/ "
                             "(features, progress, logs, IV cache, and memories); "
                             "otherwise use runs/default/")
    parser.add_argument("--no-explorer", action="store_true",
                        help="Ablation 1: random BFS subgraphs with matching hops and caps; "
                             "Synthesizer generates features and plans directly")
    parser.add_argument("--no-synthesizer", action="store_true",
                        help="Ablation 2: generate plans with hypotheses in one LLM call, without "
                             "in-loop validation, AutoFix, or consistency diagnosis; failures are rejected")
    parser.add_argument("--no-evaluator", action="store_true",
                        help="Ablation 3: random offline gating with the same budget and pool; "
                             "disable IV prediction, feedback, and evaluator memory")
    parser.add_argument("--no-diversity-guide", action="store_true",
                        help="Ablation 4: disable paradigm coverage guidance and saturation constraints")
    parser.add_argument("--no-evolution", action="store_true",
                        help="Base-Agent: disable offline evolution and evaluator RAG updates; "
                             "clear memory idempotently at episode boundaries")
    parser.add_argument("--static-skills", type=str, default=None, metavar="DIR",
                        help="Static-Agent: load frozen playbooks and evaluator memory from DIR "
                             "(config/static_agent/ layout); never update them; "
                             "retrieval uses --retriever-mode")
    parser.add_argument("--valid-target", type=int, default=None,
                        help="Stop after N admitted features; omit to run the fixed number of episodes")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Episode limit for target-based stopping (default 20)")
    parser.add_argument("--episode-resume", action="store_true",
                        help="Resume from the last episode using progress.json and "
                             "pool_checkpoint.json/features.json instead of replaying episode 0. "
                             "Disabled by default: memories and IV caches are preserved "
                             "but the online loop is replayed")
    parser.add_argument("--log", type=str, default=None,
                        help="Log to file and console; default runs/default/run.log "
                             "or runs/<run_id>/run.log with --run-id; "
                             "use 'off' to disable file logging")
    parser.add_argument("--no-clear-log", action="store_true",
                        help="Append to the previous log instead of clearing it at startup")
    parser.add_argument("--query-mode", choices=("binding", "inline"), default="binding",
                        help="binding: parameterized templates (default), keeping script text "
                             "dependent only on plan structure and passing vids as bindings "
                             "to protect the server cache; inline: legacy literals for rollback")
    args = parser.parse_args(argv)
    if args.data and args.input:
        parser.error("--data and --input are mutually exclusive")
    if args.random_seed is None:
        args.random_seed = 42 if args.data else 20260825
    if args.data and args.targets is None:
        args.targets = str(Path(args.data).expanduser().resolve()
                           / "splits" / f"target_streams_s{args.random_seed}.json")
    run_dir = None
    if args.run_id:
        run_dir = ensure_dir(RUNS_DIR / args.run_id)
    if args.log != "off":
        from .utils import install_file_logger
        log_path = args.log or ((run_dir / "run.log") if run_dir else
                                RUNS_DIR / "default" / "run.log")

        if not args.no_clear_log:
            _p = Path(log_path)
            if _p.exists():
                try:
                    _p.unlink()
                except OSError:
                    pass
        install_file_logger(log_path)
    from .execution import eg_plan as _eg_plan
    _eg_plan.set_query_mode(args.query_mode)
    node_budgets = [int(x) for x in str(args.node_budgets).split(",")
                    if x.strip().isdigit()] or None
    per_hop_caps = [int(x) for x in str(args.per_hop_caps).split(",") if x.strip()]
    max_hops = args.max_hops or (len(per_hop_caps)
                                 if args.explore_live or args.data else args.max_refine)
    persist_memory = not args.no_persist_memory
    reset_memory = args.reset_memory
    if (reset_memory or args.reset_skills or args.reset_eval_memory
            or args.reset_synth_skills) and not args.run_id \
            and not (args.skills_path or args.eval_memory_path
                     or args.synth_skills_path):
        raise SystemExit("[Reset guard] --reset-memory/--reset-* require run-scoped memory: "
                         "provide --run-id (runs/<run_id>/memory/) or explicit "
                         "--skills-path/--eval-memory-path/--synth-skills-path; "
                         "refusing to clear shared data/memory/")
    logger.info("Starting EvoFeat fraud-graph feature mining with self-evolving agents ...")
    summary = run(
        num_episodes=args.episodes,
        episode_size=args.episode_size,
        m_episodes=args.m_episodes,
        max_subgraphs=args.max_subgraphs,
        max_repair=args.max_repair,
        max_hops=max_hops,
        node_budgets=node_budgets,
        explore_topup=args.explore_topup,
        strict_depth=args.strict_depth,
        global_review=not args.no_global_review,
        skills_path=args.skills_path,
        persist_skills=persist_memory and not args.no_skill_memory,
        reset_skills=reset_memory or args.reset_skills,
        synth_skills_path=args.synth_skills_path,
        persist_synth=persist_memory and not args.no_synth_skill_memory,
        reset_synth=reset_memory or args.reset_synth_skills,
        eval_memory_path=args.eval_memory_path,
        eval_seeds_path=args.eval_seeds_path,
        persist_eval=persist_memory and not args.no_eval_memory,
        reset_eval=reset_memory or args.reset_eval_memory,
        input_path=args.input,
        qpm=args.qpm,
        max_concurrency=args.concurrency,
        threshold_mode=args.threshold_mode,
        retriever_mode=args.retriever_mode,
        semantic_check=not args.no_semantic_check,
        eg_exec=args.eg_exec or args.explore_live or bool(args.data),
        explore_live=args.explore_live or bool(args.data),
        data_dir=args.data,
        offline_panel=args.offline_panel,
        targets_path=args.targets,
        account_type=args.account_type,
        target_sample=args.target_sample,
        random_seed=args.random_seed,
        per_hop_caps=per_hop_caps,
        cold_start=args.cold_start,
        cold_start_reset=args.cold_start_reset,
        enrich_skills=args.enrich_skills,
        run_id=args.run_id,
        no_explorer=args.no_explorer,
        no_synthesizer=args.no_synthesizer,
        no_evaluator=args.no_evaluator,
        no_diversity_guide=args.no_diversity_guide,
        no_evolution=args.no_evolution,
        static_skills=args.static_skills,
        valid_target=args.valid_target,
        max_episodes=args.max_episodes,
        episode_resume=args.episode_resume,
        verbose=not args.quiet,
    )
    out_dir = run_dir or ensure_dir(RUNS_DIR / "default")
    out_path = out_dir / "run_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    features_path = out_dir / "features.json"
    features_out = summary.get("features_export",
                               {"episode_stats": [], "features": []})
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(features_out, f, ensure_ascii=False, indent=2, default=str)
    logger.info("=" * 60)
    logger.info(f"Final feature pool size: {summary['final_pool_size']}")
    logger.info(f"Feature pool: {summary['final_pool']}")
    logger.info(f"Learned skills: {len(summary['skills'])}")
    logger.info(f"Skill memory: {summary.get('skills_store') or '(not persisted)'}"
                f"   Exploration node budgets: {summary.get('explorer_budgets')}")
    logger.info(f"Evaluator memory: {summary['eval_mem_size']}  Total LLM calls: {summary['llm_calls']}")
    _rl = summary.get("rate_limit") or {}
    logger.info(f"Concurrency: {summary.get('concurrency')}  QPM limit: {_rl.get('qpm_limit') or 'unlimited'}  "
                f"Rate-limit wait: {_rl.get('wait_total_sec')}s (maximum {_rl.get('wait_max_sec')}s)")
    logger.info(f"Memory paths -> "
                f"skills: {summary.get('skills_store') or '(not persisted)'}; "
                f"evaluator: {summary.get('eval_memory_store') or '(not persisted)'}")
    logger.info(f"Run summary saved: {out_path}")
    logger.info(f"Features with verifiable EasyGraph queries saved: {features_path}")
    return summary


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
