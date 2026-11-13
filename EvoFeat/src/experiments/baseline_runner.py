from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from system.paths import PROJECT_ROOT, SOURCE_ROOT, SYSTEM_ROOT
WORKSPACE = PROJECT_ROOT

from system.utils import (ensure_dir, get_logger, install_file_logger,
                          load_config)
from system.llm.client import LLMClient
from system.core.graph import Subgraph
from system.core.schema import Candidate, TraceStep
from system.core.vocabulary import public_relation
from system.orchestrator import loop as loop_mod
from system.evolution.reward import evaluate_batch
from system.parallel import run_parallel
from experiments.logic2plan import extract_features, validate_self_plan

logger = get_logger("experiments.baseline_runner")

PE_BASELINE_DIR = os.environ.get("PE_BASELINE_DIR",
                                 str(SOURCE_ROOT / "PE_Baseline"))
_BATCH_TARGETS = 4


def _eng_of(sg) -> dict:
    hop_of = {sid: int((nd or {}).get("first_hop", 0))
              for sid, nd in sg.nodes.items()}
    nodes = [{"vid": sg.vid_of.get(sid, sid),
              "entity": (nd or {}).get("entity", "unknown"),
              "first_hop": (nd or {}).get("first_hop", 0),
              "attributes": (nd or {}).get("attributes", {}) or {}}
             for sid, nd in sg.nodes.items()]
    edges = [{"src": sg.vid_of.get(e.get("src"), e.get("src")),
              "dst": sg.vid_of.get(e.get("dst"), e.get("dst")),
              "first_hop": max(hop_of.get(e.get("src"), 0),
                               hop_of.get(e.get("dst"), 0)),
              "join_field": e.get("join_field") or e.get("jf") or "",
              "total_amount": e.get("total_amount"),
              "trade_count": e.get("trade_count"),
              "top_k": e.get("top_k")}
             for e in (sg.edges or [])]
    rel_set = sorted({e["join_field"] for e in edges if e["join_field"]})
    stats = {"hops": max(hop_of.values(), default=0), "direction": "BOTH",
             "node_count": len(nodes), "edge_count": len(edges),
             "rel_set": rel_set}
    return {"stats": stats, "nodes": nodes, "edges": edges}


def _build_b1_pre(eng: dict, idx: int) -> dict | None:
    try:
        if PE_BASELINE_DIR not in sys.path:
            sys.path.insert(0, PE_BASELINE_DIR)
        import manual_llm as afe
        bm = getattr(afe, "_orig_bm", None) or afe.build_mappings
        pa = getattr(afe, "_orig_pa", None) or afe.prune_and_align_en_subgraph
        res = {**_pe_resources(), "index": idx}
        target_vid = next((n.get("vid") for n in (eng.get("nodes") or [])
                           if int(n.get("first_hop") or 0) == 0), None)
        if not target_vid:
            return None
        entity_meta, attr_clean, rel_map, entity_map, attr_map = bm(eng, res)
        mapping_llm = {"entity_mapping": entity_map, "relation_mapping": rel_map,
                       "attribute_mapping": attr_map,
                       "prefix_legend": dict(afe.PREFIX_LEGEND)}
        subgraph_anon, vid2short, eng_pruned, prune_info = pa(
            eng, target_vid, res, entity_meta, attr_clean, mapping_llm,
            token_budget=afe.TARGET_SUBGRAPH_TOKENS)
        mapping_full = dict(mapping_llm)
        mapping_full["anonymization_mapping"] = {s: v for v, s in (vid2short or {}).items()}
        pre = {"mappings": mapping_full,
               "anonymized_english_subgraph": subgraph_anon,
               "prune_info": prune_info,
               "mapping_parts": [entity_meta, attr_clean, rel_map,
                                 entity_map, attr_map],
               "vid2short": vid2short, "eng_pruned": eng_pruned}
        json.dumps(pre, ensure_ascii=False)
        return pre
    except Exception as e:
        logger.warning(f"[baseline] Manual preprocessing failed (recompute at runtime): "
                       f"{type(e).__name__}: {e}")
        return None


def _build_b0_pre(eng: dict) -> dict | None:
    try:
        if PE_BASELINE_DIR not in sys.path:
            sys.path.insert(0, PE_BASELINE_DIR)
        import stat_llm as arm
        import comm_llm as arg_
        out = {}
        for key, mod in (("stat", arm), ("comm", arg_)):
            prune = getattr(mod, "_orig_prune", None) or mod.prune_subgraph
            out[key] = prune(copy.deepcopy(eng), mod.RELATION_IMPORTANCE)
        json.dumps(out, ensure_ascii=False)
        return out
    except Exception as e:
        logger.warning(f"[baseline] B0 preprocessing failed (recompute at runtime): "
                       f"{type(e).__name__}: {e}")
        return None


def _full_expand(source, sg, max_hops: int) -> None:
    source._fetch_neighborhood(sg, sg.center_vid)
    for d in range(2, max_hops + 1):
        pend = [c for c in source._cands.values()
                if not c["materialized"] and not c["dead"] and c["depth"] == d - 1]
        if pend:
            source._materialize(sg, [c["vid"] for c in pend])
        frontier = [c for c in source._cands.values()
                    if c["materialized"] and not c["dead"]
                    and c["depth"] == d - 1 and c["vid"] not in source._fetched]
        if not frontier:
            break
        for c in frontier:
            source._fetch_neighborhood(sg, c["vid"])
    source.finalize(sg, set(sg.nodes.keys()))


SNAP_FMT = "baseline_subgraph_v2_en"


def _snap_dir(account_type: str, seed: int) -> Path:
    return (WORKSPACE / "data" / "baseline_subgraphs"
            / f"{account_type}__s{seed}")


def _prune_to_budget(sg: Subgraph, caps: list[int], edge_limit: int) -> tuple[int, int]:
    by_hop: dict[int, list[str]] = {}
    for sid, nd in sg.nodes.items():
        by_hop.setdefault(int((nd or {}).get("first_hop", 0)), []).append(sid)
    kept = set(by_hop.get(0, []))
    for h in sorted(k for k in by_hop if k >= 1):
        cap = caps[h - 1] if h - 1 < len(caps) else (caps[-1] if caps else 0)
        kept.update(sorted(by_hop[h])[:cap])
    dropped_n = len(sg.nodes) - len(kept)
    if dropped_n > 0:
        sg.nodes = {sid: nd for sid, nd in sg.nodes.items() if sid in kept}
        sg.edges = [e for e in sg.edges
                    if e.get("src") in kept and e.get("dst") in kept]
        sg.vid_of = {k: v for k, v in sg.vid_of.items() if k in kept}
    dropped_e = 0
    if edge_limit and len(sg.edges) > edge_limit:
        dropped_e = len(sg.edges) - edge_limit
        sg.edges = sg.edges[:edge_limit]
    if dropped_n or dropped_e:
        sg._build_adj()
        logger.info(f"[baseline] Subgraph pruned to budget: nodes -{dropped_n}, edges -{dropped_e}"
                    f" (caps={caps}, edge_limit={edge_limit})")
    return dropped_n, dropped_e


def _save_subgraph(path: Path, sg: Subgraph, caps: list[int], edge_limit: int,
                   meta: dict, extra: dict | None = None) -> None:
    payload = {"format": SNAP_FMT, "kind": SNAP_FMT, "caps": list(caps), "edge_limit": edge_limit,
               "subgraph_id": sg.subgraph_id, "center": sg.center,
               "center_vid": sg.center_vid, "nodes": sg.nodes,
               "edges": sg.edges, "vid_of": sg.vid_of, **meta}
    for k, v in (extra or {}).items():
        if v is not None:
            payload[k] = v
    ensure_dir(path.parent)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _load_subgraph(path: Path, caps: list[int], edge_limit: int) -> tuple:
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("format") != SNAP_FMT or d.get("kind") != SNAP_FMT:
        raise ValueError(f"[baseline] Subgraph snapshot format mismatch ({path.name}): "
                         f"format={d.get('format')}, kind={d.get('kind')}")
    sg = Subgraph(subgraph_id=d["subgraph_id"], center=d["center"],
                  center_vid=d["center_vid"], nodes=d["nodes"],
                  edges=d["edges"], vid_of=d["vid_of"])
    sg._build_adj()
    _prune_to_budget(sg, caps, edge_limit)
    return sg, d


_CKPT_FMT = "baseline_target_v2_en"
_TARGETS_DIRNAME = "targets"
_CAND_TRANSIENT = (
    "_death_stage", "_baseline", "_self_plan", "_skills_used", "_err_history",
    "_failed_plan", "_synth_trace", "_stratum",
    "_explorer_fault", "_accounted_dead",
)


def _targets_dir(run_dir) -> Path:
    return Path(run_dir) / _TARGETS_DIRNAME


def _target_ckpt_path(run_dir, seq: int) -> Path:
    return _targets_dir(run_dir) / f"seq_{seq:03d}.json"


def _trace_to_json(t) -> dict:
    if hasattr(t, "to_dict"):
        d = dict(t.to_dict())
    elif isinstance(t, dict):
        d = dict(t)
    else:
        return {"action": str(t)}
    ev = getattr(t, "evidence", None) or d.get("evidence")
    if ev:
        d["evidence"] = list(ev)
    return d


def _cand_to_json(c: Candidate) -> dict:
    d = c.to_dict()
    d["_trace"] = [_trace_to_json(t) for t in (getattr(c, "trace", None) or [])]
    d["_transient"] = {k: getattr(c, k) for k in _CAND_TRANSIENT if hasattr(c, k)}
    return d


def _cand_from_json(d: dict) -> Candidate:
    c = Candidate.from_dict({k: v for k, v in d.items() if not k.startswith("_")})
    c.trace = [TraceStep(action=str(x.get("action") or ""),
                         obs=str(x.get("obs") or ""),
                         thought=str(x.get("thought") or ""),
                         evidence=list(x.get("evidence") or []))
               for x in (d.get("_trace") or []) if isinstance(x, dict)]
    for k, v in (d.get("_transient") or {}).items():
        setattr(c, k, v)

    for k in ("consistency_detail",):
        if k in d and d[k] is None:
            setattr(c, k, None)
    return c


def _save_target_ckpt(run_dir, seq: int, meta: dict, center_vid: str,
                      cands: list) -> None:
    path = _target_ckpt_path(run_dir, seq)
    ensure_dir(path.parent)
    payload = {"format": _CKPT_FMT, "kind": _CKPT_FMT, "seq": seq, "center_vid": center_vid,
               **meta, "cands": [_cand_to_json(c) for c in cands]}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, default=str),
                   encoding="utf-8")
    tmp.replace(path)


def _load_target_ckpt(run_dir, seq: int, meta: dict,
                      center_vid: "str | None") -> "list | None":
    path = _target_ckpt_path(run_dir, seq)
    if not path.is_file():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"[resume] Target {seq} cache parsing failed; rerunning: {e}")
        return None
    if d.get("format") != _CKPT_FMT or d.get("kind") != _CKPT_FMT:
        logger.warning(f"[resume] Target {seq} cache format mismatch "
                       f"(format={d.get('format')}, kind={d.get('kind')}); rerunning")
        return None
    for k, v in meta.items():
        if d.get(k) != v:
            logger.warning(f"[resume] Target {seq} cached {k}={d.get(k)!r} != current {v!r}; "
                           f"rerunning")
            return None
    if center_vid and d.get("center_vid") and d["center_vid"] != center_vid:
        logger.warning(f"[resume] Target {seq} cached anchor {d['center_vid']} != target stream "
                       f"{center_vid} (pairing mismatch); rerunning")
        return None
    try:
        return [_cand_from_json(x) for x in (d.get("cands") or [])]
    except Exception as e:
        logger.warning(f"[resume] Target {seq} candidate deserialization failed; rerunning: {e}")
        return None


_PRUNE_AUDIT: list = []


def _wrap_prune_pairs(mod, engine: str) -> None:
    if getattr(mod, "_audit_wrapped", False):
        return
    orig_prune = mod.prune_subgraph
    mod._orig_prune = orig_prune

    def wrapped(sg, rel):
        orig_tok = mod.estimate_subgraph_tokens(sg)
        pre = _PRE.get("cur")
        if pre is not None:
            _PRUNE_AUDIT.append({
                "engine": engine,
                "orig_tokens": orig_tok,
                "final_tokens": mod.estimate_subgraph_tokens(pre),
                "orig_nodes": len(sg.get("nodes", [])),
                "final_nodes": len(pre.get("nodes", [])),
                "budget": mod.TARGET_SUBGRAPH_TOKENS,
                "precomputed": True,
            })
            return copy.deepcopy(pre)
        out = orig_prune(sg, rel)
        _PRUNE_AUDIT.append({
            "engine": engine,
            "orig_tokens": orig_tok,
            "final_tokens": mod.estimate_subgraph_tokens(out),
            "orig_nodes": len(sg.get("nodes", [])),
            "final_nodes": len(out.get("nodes", [])),
            "budget": mod.TARGET_SUBGRAPH_TOKENS,
            "precomputed": False,
        })
        return out

    mod.prune_subgraph = wrapped
    mod._audit_wrapped = True


_PRE: dict = {"cur": None}


def _audit_b1(info: dict, precomputed: bool) -> None:
    _PRUNE_AUDIT.append({
        "engine": "manual",
        "orig_tokens": info.get("orig_tokens"),
        "final_tokens": info.get("final_tokens"),
        "orig_nodes": info.get("orig_nodes"),
        "final_nodes": info.get("kept_nodes"),
        "mapping_tokens": info.get("mapping_tokens"),
        "budget": info.get("subgraph_budget"),
        "precomputed": precomputed,
    })


def _wrap_prune_b1(afe) -> None:
    if getattr(afe, "_audit_wrapped", False):
        return
    orig_pa = afe.prune_and_align_en_subgraph
    orig_bm = afe.build_mappings
    afe._orig_pa = orig_pa
    afe._orig_bm = orig_bm

    def wrapped_pa(*a, **kw):
        pre = _PRE.get("cur")
        if pre is not None:
            _audit_b1(pre.get("prune_info") or {}, precomputed=True)
            return (pre["anonymized_english_subgraph"], pre["vid2short"],
                    pre["eng_pruned"], pre["prune_info"])
        res = orig_pa(*a, **kw)
        info = res[3] if isinstance(res, tuple) and len(res) > 3 else {}
        _audit_b1(info, precomputed=False)
        return res

    def wrapped_bm(eng, res):
        pre = _PRE.get("cur")
        if pre is not None:
            return tuple(pre["mapping_parts"])
        return orig_bm(eng, res)

    afe.prune_and_align_en_subgraph = wrapped_pa
    afe.build_mappings = wrapped_bm
    afe._audit_wrapped = True


def _import_baseline(name: str):
    sys.path.insert(0, PE_BASELINE_DIR)
    try:
        if name == "manual":
            import manual_llm as afe
            _wrap_prune_b1(afe)
            def _run(eng, idx, pre=None):
                _PRE["cur"] = pre
                try:
                    return afe.process_subgraph_en(
                        eng, idx, model=_model(),
                        res={**_pe_resources(), "index": idx})
                finally:
                    _PRE["cur"] = None
            return _run
        if name == "stat":
            import stat_llm as arm
            _wrap_prune_pairs(arm, "stat")
            def _run(eng, idx, pre=None):
                _PRE["cur"] = pre
                try:
                    return arm.process_single_subgraph(eng, idx, model=_model())
                finally:
                    _PRE["cur"] = None
            return _run
        if name == "comm":
            import comm_llm as arg_
            _wrap_prune_pairs(arg_, "comm")
            def _run(eng, idx, pre=None):
                _PRE["cur"] = pre
                try:
                    return arg_.process_subgraph_graphrag(eng, idx, model=_model())
                finally:
                    _PRE["cur"] = None
            return _run
    except ImportError as e:
        raise SystemExit(f"[baseline_runner] PE_Baseline import failed ({e}); "
                         f"check PE_BASELINE_DIR (currently {PE_BASELINE_DIR})")
    raise SystemExit(f"[baseline_runner] Unknown baseline {name}")


_ENTITY_NAMES_CACHE: list | None = None
_PE_RES_CACHE: dict | None = None


def _entity_names() -> list:
    global _ENTITY_NAMES_CACHE
    if _ENTITY_NAMES_CACHE is None:
        from system.core.graph import entity_index
        _ENTITY_NAMES_CACHE = list(entity_index().keys())
    return _ENTITY_NAMES_CACHE


def _pe_resources() -> dict:
    global _PE_RES_CACHE
    if _PE_RES_CACHE is not None:
        return _PE_RES_CACHE
    if PE_BASELINE_DIR not in sys.path:
        sys.path.insert(0, PE_BASELINE_DIR)
    import manual_llm as afe
    res = dict(afe.load_resources())
    res["entity_names"] = sorted(set(res.get("entity_names") or [])
                                 | set(_entity_names()))
    _PE_RES_CACHE = res
    return res


_VTYPE_CLEANER = None


def _vtype_blocked(entity: str) -> bool:
    global _VTYPE_CLEANER
    if _VTYPE_CLEANER is None:
        from system.core.graph_clean import GraphCleaner
        exec_cfg = load_config("execution.yaml")
        live_cfg = exec_cfg.get("explore_live") or {}
        bl_cfg = live_cfg.get("blacklist") or {}
        res_dir = SYSTEM_ROOT / "data" / "graph_res"
        _VTYPE_CLEANER = GraphCleaner(
            vtype_blacklist_path=(bl_cfg.get("vtype")
                                  or str(res_dir / "vtype_blacklist.txt")),
            vid_blacklist_path=None, field_blacklist_path=None)
    return _VTYPE_CLEANER.vtype_blocked(entity)


def _drop_blocked_eng(eng: dict) -> tuple[int, int]:
    nodes = eng.get("nodes") or []
    edges = eng.get("edges") or []
    blocked = {str(n.get("vid") or "") for n in nodes
               if _vtype_blocked(str(n.get("entity") or ""))}
    kept_nodes = [n for n in nodes if str(n.get("vid") or "") not in blocked]
    kept_edges = [e for e in edges
                  if str(e.get("src") or "") not in blocked
                  and str(e.get("dst") or "") not in blocked
                  and public_relation(e.get("join_field") or e.get("relation") or "")]
    eng["nodes"], eng["edges"] = kept_nodes, kept_edges
    stats = eng.get("stats")
    if isinstance(stats, dict):
        stats["node_count"] = len(kept_nodes)
        stats["edge_count"] = len(kept_edges)
        stats["rel_set"] = sorted({e.get("join_field") for e in kept_edges
                                   if e.get("join_field")})
        hop_vals = [int(n.get("first_hop") or 0) for n in kept_nodes]
        if hop_vals:
            stats["hops"] = max(hop_vals)
    return len(nodes) - len(kept_nodes), len(edges) - len(kept_edges)


def _inputs_of(baseline: str, payload: dict | None, sg, seq: int):
    eng = (payload or {}).get("eng") or _eng_of(sg)
    n_rm, e_rm = _drop_blocked_eng(eng)
    b1_pre = (payload or {}).get("b1_pre")
    if n_rm or e_rm:
        logger.info(f"[baseline {baseline}] Target {seq}: non-public entities or relations filtered from PE input"
                    f" (nodes -{n_rm}, edges -{e_rm}; derived inputs recomputed from the filtered subgraph)")
        b1_pre = None
    if baseline == "manual":
        return eng, (b1_pre if b1_pre is not None else _build_b1_pre(eng, seq))
    if baseline in ("stat", "comm"):
        if b1_pre is None:
            b1_pre = _build_b1_pre(eng, seq) or {}
        anon = b1_pre.get("anonymized_english_subgraph")
        if not isinstance(anon, dict):
            return None, None
        return anon, (_build_b0_pre(anon) or {}).get(baseline)
    return eng, None


def _model() -> str:
    return os.environ.get("SYSTEM_LLM_MODEL", "hy4-preview")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, choices=["stat", "comm", "manual"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--account-type", default="individual", choices=["individual"])
    ap.add_argument("--random-seed", type=int, default=20260825)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--valid-target", type=int, default=100)
    ap.add_argument("--target-sample", type=int, default=100)
    ap.add_argument("--data", default=None, metavar="DIR",
                    help="Released FraudFeat Parquet directory; run the PE baseline locally")
    ap.add_argument("--offline-panel", default=None,
                    help="Fixed offline IV panel (default <data>/splits/offline_panel.parquet)")
    ap.add_argument("--targets", default=None, help="Optional prepared public target JSON")
    ap.add_argument("--log", type=str, default=None,
                    help="Write logs to a file and the console; "
                         "default: runs/<run_id>/run.log; use 'off' to disable file logging")
    ap.add_argument("--no-clear-log", action="store_true",
                    help="Append to the existing log file instead of clearing it (default: overwrite)")
    ap.add_argument("--no-resume", action="store_true",
                    help="Disable target-level resume: do not read or write runs/<run-id>/targets/; "
                         "rerun all targets (default: cached targets skip PE analysis and validation)")
    ap.add_argument("--query-mode", choices=("binding", "inline"), default="binding",
                    help="Query mode (must match system.main semantics and defaults for fair RQ6 cost comparison): "
                         "binding=parameterized templates (default) / "
                         "inline=legacy inline literals (rollback)")
    args = ap.parse_args()
    from system.selfcheck import verify as verify_deps
    verify_deps(f"baseline {args.baseline}")
    from system.execution import eg_plan as _eg_plan
    _eg_plan.set_query_mode(args.query_mode)
    out_dir = ensure_dir(WORKSPACE / "runs" / args.run_id)
    os.environ["IV_RESULTS_DIR"] = str(out_dir / "iv_values")
    process = _import_baseline(args.baseline)

    if args.log != "off":
        log_path = Path(args.log) if args.log else (out_dir / "run.log")
        if log_path.exists() and not args.no_clear_log:
            log_path.unlink()
        install_file_logger(log_path)
        logger.info(f"[baseline {args.baseline}] Run log -> {log_path}")
    from system.execution import build_executor_from_config, preflight
    from system.execution.graph_source import (LiveGraphSource, ParquetGraphSource,
                                               ParquetTargetStream)
    from system.core.graph_clean import GraphCleaner
    from system.core.graph import entity_index
    if args.data:
        from system.execution.parquet_backend import ParquetExecutor
        data_dir = Path(args.data).expanduser().resolve()
        executor = ParquetExecutor(
            data_dir, panel_path=args.offline_panel
            or data_dir / "splits" / "offline_panel.parquet")
    else:
        from system.data.target_stream import TargetStream
        executor = build_executor_from_config(enabled=True)
    if executor is None:
        raise SystemExit("[baseline_runner] Graph executor unavailable")
    preflight(executor)
    exec_cfg = load_config("execution.yaml")
    live_cfg = exec_cfg.get("explore_live") or {}
    bl_cfg = live_cfg.get("blacklist") or {}
    res_dir = SYSTEM_ROOT / "data" / "graph_res"
    cleaner = GraphCleaner(
        vtype_blacklist_path=bl_cfg.get("vtype") or str(res_dir / "vtype_blacklist.txt"),
        vid_blacklist_path=bl_cfg.get("vid") or str(res_dir / "vid_blacklist.txt"),
        field_blacklist_path=bl_cfg.get("edge_field") or str(res_dir / "edge_field_blacklist.txt"))
    caps = [int(x) for x in str(live_cfg.get("per_hop_caps", "50,250,500,1000,2000")
                                 ).split(",") if x.strip()]
    edge_limit = int(live_cfg.get("edge_limit", 5000))
    source_cls = ParquetGraphSource if args.data else LiveGraphSource
    source = source_cls(
        executor, cleaner, per_hop_caps=caps,
        edge_limit=edge_limit,
        min_hops=int(live_cfg.get("min_hops", 3)),
        probe_min_hops=int(live_cfg.get("probe_min_hops",
                                        live_cfg.get("min_hops", 3))),
        probe_max_frontier=int(live_cfg.get("probe_max_frontier", 8)),
        induced_edges=bool(live_cfg.get("induced_edges", True)),
        probe_quiet=bool(live_cfg.get("probe_quiet", True)),
        entity_names=list(entity_index().keys()))
    if args.data:
        stream = ParquetTargetStream(source, sample=args.target_sample,
                                     seed=args.random_seed, targets_path=args.targets)
    else:
        stream = TargetStream(source, sr_cfg=exec_cfg.get("starrocks") or {},
                              account_type=args.account_type, sample=args.target_sample,
                              seed=args.random_seed,
                              cache_cfg=live_cfg.get("target_cache") or {})
    llm_consistency = LLMClient(name="consistency")
    pool: list = []
    history: list[dict] = []
    seq = 0
    snap_dir = _snap_dir(args.account_type, args.random_seed)
    resume = not args.no_resume
    ckpt_meta = {"baseline": args.baseline, "account_type": args.account_type,
                 "seed": args.random_seed, "target_sample": args.target_sample}
    if resume:
        n_ckpt = len(list(_targets_dir(out_dir).glob("seq_*.json")))
        if n_ckpt:
            logger.info(f"[resume] Found {n_ckpt} target caches ({_targets_dir(out_dir)}): "
                        f"cached targets skip PE analysis and validation (use --no-resume to rerun all)")
    else:
        logger.info("[resume] --no-resume: no target cache reads or writes; rerunning all targets")
    while seq < args.target_sample:
        sg = stream.next()
        if sg is None:
            break
        cur_vid = sg.center_vid
        snap = snap_dir / f"seq_{seq + 1:03d}.json"
        if snap.is_file():
            seq += 1
            sg, payload = _load_subgraph(snap, caps, edge_limit)
            logger.info(f"[baseline {args.baseline}] Target {seq}: reusing local subgraph snapshot"
                        f" ({snap.name}, nodes {len(sg.nodes)} edges {len(sg.edges)})")
        else:
            if source.probe_min_hops > 0 and not source.prepare(sg, None):
                logger.info(f"[baseline {args.baseline}] Target depth below "
                            f"{source.probe_min_hops} hops; skipping (not counted as a target)")
                continue
            seq += 1
            _full_expand(source, sg, len(caps))
            dn, de = _prune_to_budget(sg, caps, edge_limit)
            eng = _eng_of(sg)
            b1_pre = _build_b1_pre(eng, seq) if args.baseline == "manual" else None
            _save_subgraph(snap, sg, caps, edge_limit,
                           {"account_type": args.account_type,
                            "seed": args.random_seed, "seq": seq,
                            "pruned_nodes": dn, "pruned_edges": de},
                           extra={"eng": eng, "b1_pre": b1_pre})
            payload = {"eng": eng, "b1_pre": b1_pre}
            logger.info(f"[baseline {args.baseline}] Target {seq}: subgraph expanded and snapshot saved"
                        f" ({snap.name}, nodes {len(sg.nodes)} edges {len(sg.edges)})")
        episode = (seq - 1) // _BATCH_TARGETS
        ckpt = (_load_target_ckpt(out_dir, seq, ckpt_meta, cur_vid)
                if resume else None)
        if ckpt is not None:
            batch_cands = ckpt
            logger.info(f"[resume] Target {seq}: cache hit ({len(ckpt)} candidates, "
                        f"{sum(1 for c in ckpt if c.synth_ok)} surviving); skipping analysis and validation")
        else:
            inp, pre = _inputs_of(args.baseline, payload, sg, seq)
            if inp is None:
                logger.warning(f"[baseline {args.baseline}] Target {seq}: "
                               f"anonymized English subgraph missing (b1_pre preprocessing failed); skipping")
                continue
            try:
                result = process(inp, seq, pre=pre)
            except Exception as e:
                logger.warning(f"[baseline {args.baseline}] Target {seq} analysis failed ({e}); skipping")
                continue
            feats = extract_features(result, args.baseline)

            if _PRUNE_AUDIT:
                with open(out_dir / "prune_audit.jsonl", "a", encoding="utf-8") as f:
                    for entry in _PRUNE_AUDIT:
                        entry["seq"] = seq
                        entry["episode"] = episode
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                _PRUNE_AUDIT.clear()
            logger.info(f"[baseline {args.baseline}] Target {seq}: generated {len(feats)} features")
            def _validate_one(feat: dict, _i: int):
                cand = validate_self_plan(feat, sg, executor, llm_consistency)
                cand.episode = episode
                return cand
            batch_cands = [c for c in run_parallel(
                feats, _validate_one, max_workers=args.concurrency,
                stage=f"Target {seq}: validating {len(feats)} features")
                if c is not None]
            if resume:
                _save_target_ckpt(out_dir, seq, ckpt_meta, cur_vid, batch_cands)
        alive = [c for c in batch_cands if c.synth_ok]
        if alive:
            evaluate_batch(alive, pool, executor=executor)
            pool.extend(alive)
        loop_mod._write_candidates(out_dir, batch_cands, pool, episode)
        loop_mod._write_case_materials(out_dir, batch_cands, pool, episode)
        history.append({"episode": episode, "subgraph_seq": seq,
                        "cum_valid": len(pool), "cum_subgraphs": seq,
                        "grounded": len(alive)})
        loop_mod._checkpoint_base(out_dir, pool, history)
        logger.info(f"[baseline {args.baseline}] Target {seq}: added {len(alive)} valid features to the pool; "
                    f"valid pool size {len(pool)}")
        if args.valid_target and len(pool) >= args.valid_target:
            logger.info(f"[stop rule] Valid feature count {len(pool)} >= {args.valid_target}; "
                        f"target reached, stopping ({seq} subgraphs used)")
            break
    summary = {"run_id": args.run_id, "baseline": args.baseline,
               "subgraphs_processed": seq, "final_pool_size": len(pool),
               "valid_target": args.valid_target,
               "reached": len(pool) >= args.valid_target,
               "data_dir": str(data_dir) if args.data else None,
               "offline_panel": getattr(executor, "panel_path", None),
               "targets_path": str(getattr(stream, "targets_path", "") or "") or None,
               "random_seed": args.random_seed,
               "evaluation": {"backend": "fraudfeat" if args.data else "easygraph",
                              "amount_unit": "fen", "id_list_limit": 300}}
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[baseline {args.baseline}] Complete: targets {seq}, valid features {len(pool)} "
                f"-> {out_dir}")


if __name__ == "__main__":
    main()
