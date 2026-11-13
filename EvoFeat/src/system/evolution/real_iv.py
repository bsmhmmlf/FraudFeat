from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Dict, List, Optional

import pandas as pd

from ..core.schema import Candidate
from ..execution import (
    SERVER_OUTAGE_ERR_PREFIX, SLOW_QUERY_ERR_PREFIX,
    build_executor_from_config, execute_plan,
    normalize_plan, sanitize_vid, validate_plan,
)
from ..paths import DATA_DIR, PROJECT_ROOT, SYSTEM_ROOT
from ..utils import get_logger, load_config
from .autobin import AutobinSplit


logger = get_logger("real_iv")
_CFG_CACHE: Optional[Dict[str, Any]] = None


def load_iv_config() -> Dict[str, Any]:
    global _CFG_CACHE
    if _CFG_CACHE is None:
        try:
            cfg = load_config("iv.yaml")
        except Exception:
            cfg = {}
        _CFG_CACHE = cfg if isinstance(cfg, dict) else {}
    return _CFG_CACHE


def real_mode_enabled() -> bool:
    return str(load_iv_config().get("mode", "simulate")).strip().lower() == "real"


def _resolve(p: str) -> str:
    if not p:
        return ""
    return p if os.path.isabs(p) else str(PROJECT_ROOT / p)


_PANEL_CACHE: Dict[Any, Optional[List[Dict[str, Any]]]] = {}

_PANEL_LOCK = threading.Lock()
_TARGET_VIDS_CACHE: Optional[set] = None


def _target_entry_vids() -> set:
    global _TARGET_VIDS_CACHE
    if _TARGET_VIDS_CACHE is not None:
        return _TARGET_VIDS_CACHE
    import glob as _glob
    vids: set = set()
    root = str(DATA_DIR / "target_cache")
    for fp in _glob.glob(os.path.join(root, "target_ids_*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        for ids in (data.get("ids") or {}).values():
            for v in ids:
                vid = sanitize_vid(str(v))
                if vid:
                    vids.add(vid)
    if vids:
        logger.info(f"[True IV] Panel excludes {len(vids)} mining-target entry vids (target_cache)")
    _TARGET_VIDS_CACHE = vids
    return vids


def _pick_large(out_dir: str, want: str) -> tuple:
    try:
        larges = [f for f in os.listdir(out_dir) if f.endswith("_large.json")]
    except OSError:
        larges = []
    exact = os.path.join(out_dir, want)
    if os.path.isfile(exact):
        return exact, larges
    stem = want.replace("_samples_large.json", "").replace("_large.json", "")
    hits = [f for f in larges if stem and f.startswith(stem)]
    if not hits:
        wt = [t for t in want.split(".")[0].split("_") if t]
        best: list = []
        for f in larges:
            ft = [t for t in f.split(".")[0].split("_") if t]
            run = 0
            for i in range(len(wt)):
                for j in range(len(ft)):
                    k = 0
                    while (i + k < len(wt) and j + k < len(ft)
                           and wt[i + k] == ft[j + k]):
                        k += 1
                    run = max(run, k)
            best.append((run, f))
        if best:
            top = max(best, key=lambda x: x[0])
            if top[0] >= 2:
                hits = [top[1]]
    return (os.path.join(out_dir, hits[0]) if hits else ""), larges


def _auto_collect_samples(samples_abs: str, cfg: Dict[str, Any]) -> bool:
    if not cfg.get("auto_collect", True):
        return False
    script = str(cfg.get("collector_script") or "").strip()
    if not script:
        script = str(SYSTEM_ROOT / "scripts" / "collect_samples_by_prefix.py")
    if not os.path.isfile(script):
        logger.warning(f"[True IV] Automatic collection skipped: collector script does not exist {script}")
        return False
    out_dir = os.path.join(os.path.dirname(os.path.abspath(script)), "data")
    want = os.path.basename(samples_abs)
    src, larges = _pick_large(out_dir, want)
    if src:
        logger.info(f"[True IV] Reusing collected artifact {os.path.basename(src)}"
                    f" (avoiding a full StarRocks rescan) -> copying as {want}")
    else:
        timeout = int(cfg.get("collect_timeout_s", 1800) or 1800)
        logger.info(f"[True IV] Sample file missing/unreadable; automatic collection: {os.path.basename(script)}"
                    f" (full StarRocks scan, timeout {timeout}s)...")
        try:
            proc = subprocess.run([sys.executable, script], capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning(f"[True IV] Automatic collection timed out (>{timeout}s); freezing")
            return False
        except Exception as e:
            logger.warning(f"[True IV] Failed to start automatic collection ({type(e).__name__}: {e}); freezing")
            return False
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            logger.warning(f"[True IV] Automatic collection failed rc={proc.returncode}"
                           f" ({tail[-1][:160] if tail else 'no output'}); freezing")
            return False
        src, larges = _pick_large(out_dir, want)
        if not src:
            logger.warning(f"[True IV] Automatic collection completed but artifact is missing (expected {want})"
                           f" (large files under data/={larges}); freezing")
            return False
    try:
        os.makedirs(os.path.dirname(samples_abs), exist_ok=True)
        shutil.copy2(src, samples_abs)
    except OSError as e:
        logger.warning(f"[True IV] Sample copy failed ({e}): {src} -> {samples_abs}; freezing")
        return False
    logger.info(f"[True IV] Automatic collection completed: {src} -> {samples_abs}")
    return True


def _load_panel(path: str, n_black: int, seed: int,
                cfg: Optional[Dict[str, Any]] = None,
                n_white: Optional[int] = None,
                exclude_vids: Optional[set] = None) -> Optional[List[Dict[str, Any]]]:
    for attempt in range(2):
        if not path or not os.path.isfile(path):
            reason = f"Sample file does not exist: {path or '(samples_path not configured)'}"
            data = None
        else:
            reason = ""
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    reason = f"Invalid sample-file structure (expected an array): {path}"
                    data = None
            except Exception as e:
                reason = f"Failed to read sample file ({type(e).__name__}: {e}): {path}"
                data = None
        if data is not None:
            break
        if attempt == 0 and _auto_collect_samples(path, cfg or {}):
            continue
        logger.warning(f"[True IV] {reason}")
        return None
    black: List[Dict[str, Any]] = []
    white: List[Dict[str, Any]] = []
    seen = set()
    target_vids = _target_entry_vids() | set(exclude_vids or set())
    n_excluded = 0
    for it in data:
        if not isinstance(it, dict):
            continue
        tid = sanitize_vid(str(it.get("target_id") or it.get("unique_id") or "").strip())
        tag = str(it.get("tag") or "").strip()
        if not tag and it.get("label") is not None:
            tag = "black" if str(it.get("label")).strip() in {"1", "1.0", "true"} else "white"
        if not tid or tid in seen:
            continue
        if tid in target_vids:
            n_excluded += 1
            continue
        seen.add(tid)
        rec = {"target_id": tid, "tag": tag, "label": 1 if tag == "black" else 0}
        if tag == "black":
            black.append(rec)
        elif tag == "white":
            white.append(rec)
    if not black or not white:
        logger.warning(f"[True IV] One sample class is empty (black={len(black)}, white={len(white)}): {path}")
        return None
    rng = random.Random(seed)
    n_w = int(n_white if n_white is not None else (cfg or {}).get("panel_white", 5 * n_black))
    if len(black) < n_black or len(white) < n_w:
        logger.warning(f"[True IV] Insufficient samples: required {n_black}+{n_w}, available {len(black)}+{len(white)}")
        return None
    sel_b = rng.sample(black, n_black)
    sel_w = rng.sample(white, n_w)
    panel = sel_b + sel_w
    logger.info(f"[True IV] Sample panel: black {len(sel_b)}/{len(black)} + white {len(sel_w)}/{len(white)}"
                f" (target {n_black}+{n_w}, seed={seed}, source {os.path.basename(path)}, "
                f"excluded {n_excluded} overlapping targets)")
    return panel


def _load_panel_cached(cfg: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    path = _resolve(str(cfg.get("samples_path") or ""))
    n_black = int(cfg.get("panel_black", 1000) or 1000)
    n_white = int(cfg.get("panel_white", 5 * n_black) or (5 * n_black))
    seed = int(cfg.get("random_seed", 42) or 42)
    key = (path, n_black, n_white, seed)
    with _PANEL_LOCK:
        if key not in _PANEL_CACHE:
            _PANEL_CACHE[key] = _load_panel(path, n_black, seed, cfg, n_white)
        return _PANEL_CACHE[key]


def _depth_cache_path(cfg: Dict[str, Any], min_hops: int) -> str:
    rel = str(cfg.get("depth_cache_path")
              or f"data/samples/panel_depth_cache_h{min_hops}.json")
    return _resolve(rel)


def _load_depth_cache(path: str) -> Dict[str, bool]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): bool(v) for k, v in data.items()}
    except Exception:
        return {}


def _save_depth_cache(path: str, cache: Dict[str, bool]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError as e:
        logger.warning(f"[True IV] Failed to write depth cache ({e}); probing again next time")


def _depth_filter_panel(panel: List[Dict[str, Any]],
                        cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    from ..execution import build_executor_from_config
    from ..execution.graph_source import LiveGraphSource
    _pmh = cfg.get("panel_min_hops")
    min_hops = int(_pmh) if _pmh is not None else 5
    if min_hops <= 0:
        return panel
    path = _depth_cache_path(cfg, min_hops)
    cache = _load_depth_cache(path)
    ex = build_executor_from_config(enabled=True)
    if ex is None:
        logger.warning("[True IV] Panel depth filtering skipped: executor unavailable (panel is NOT depth-filtered!)")
        return panel
    src = LiveGraphSource(ex, min_hops=min_hops, probe_min_hops=min_hops,
                          probe_quiet=True)
    kept: List[Dict[str, Any]] = []
    n_probe = n_drop = 0
    try:
        for rec in panel:
            vid = str(rec.get("target_id") or "")
            ok = cache.get(vid)
            if ok is None:
                sg = src.init_target(vid)
                ok = bool(sg is not None and src.prepare(sg, None))
                cache[vid] = ok
                n_probe += 1
                if n_probe % 50 == 0:
                    _save_depth_cache(path, cache)
            if ok:
                kept.append(rec)
            else:
                n_drop += 1
    finally:
        _save_depth_cache(path, cache)
        close = getattr(ex, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    logger.info(f"[True IV] Panel depth filter (>={min_hops} hops): {len(panel)} -> {len(kept)}"
                f" (new probes {n_probe}, dropped {n_drop}, cached {len(cache)} entries)")
    if len(kept) < len(panel):
        logger.warning(f"[True IV] Panel reduced by {len(panel) - len(kept)} entries after depth filtering"
                       f" (pool of 5000 black/10000 white greatly exceeds panel requirements; check sample depth if reduction is excessive)")
    return kept


class _RunnerPool:
    def __init__(self, factory):
        self._factory = factory
        self._idle: List[Any] = []
        self._all: List[Any] = []
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            if self._idle:
                return self._idle.pop()
        r = self._factory()
        if r is None:
            raise RuntimeError("Failed to build EasyGraph executor (check execution.yaml / environment variables)")
        with self._lock:
            self._all.append(r)
        return r

    def release(self, r) -> None:
        with self._lock:
            self._idle.append(r)


_POOL: Optional[_RunnerPool] = None
_POOL_LOCK = threading.Lock()


def _get_pool(executor: Any = None) -> _RunnerPool:
    if executor is not None:
        return _RunnerPool(lambda: executor)
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = _RunnerPool(lambda: build_executor_from_config(enabled=True))
    return _POOL


_ENV_FATAL_MARKERS = (
    "easygraph-v3-client-python is not installed", "not in a graph-enabled environment", "Polaris service discovery is unavailable",
    "easygraph-v3-client",
)


def _env_fatal(runner: Any) -> str:
    if getattr(runner, "backend", "") == "fraudfeat-parquet":
        return ""
    try:
        runner.submit("__env_probe__", {})
    except Exception as e:
        s = f"{type(e).__name__}: {e}"
        if any(m in s for m in _ENV_FATAL_MARKERS):
            return s
    return ""


class _OutageAbort(Exception):
    pass


def _query_one(plan: Dict[str, Any], target: Dict[str, Any], pool: _RunnerPool) -> Dict[str, Any]:
    runner = pool.acquire()
    try:
        value, _trace, err = execute_plan(runner, plan, target["target_id"])
    finally:
        pool.release(runner)
    if err:
        if err.startswith(SERVER_OUTAGE_ERR_PREFIX):
            raise _OutageAbort(err)
        kind = "slow_query" if err.startswith(SLOW_QUERY_ERR_PREFIX) else "query_error"
        return {"target_id": target["target_id"], "value": None,
                "tag": target["tag"], "error": kind}
    return {"target_id": target["target_id"], "value": value,
            "tag": target["tag"], "error": ""}


_SILENCE_LOCK = threading.Lock()
_SILENCE_DEPTH = 0
_SILENCE_SAVED: List[int] = []


def _exec_loggers() -> List[logging.Logger]:
    return [logging.getLogger("system.execution.plan"),
            logging.getLogger("system.execution.eg")]


def _silence_exec_logs() -> None:
    global _SILENCE_DEPTH, _SILENCE_SAVED
    with _SILENCE_LOCK:
        if _SILENCE_DEPTH == 0:
            lgs = _exec_loggers()
            _SILENCE_SAVED = [lg.level for lg in lgs]
            for lg in lgs:
                lg.setLevel(logging.WARNING)
        _SILENCE_DEPTH += 1


def _restore_exec_logs() -> None:
    global _SILENCE_DEPTH
    with _SILENCE_LOCK:
        _SILENCE_DEPTH = max(0, _SILENCE_DEPTH - 1)
        if _SILENCE_DEPTH == 0 and _SILENCE_SAVED:
            for lg, lv in zip(_exec_loggers(), _SILENCE_SAVED):
                lg.setLevel(lv)


def _batch_query(plan: Dict[str, Any], panel: List[Dict[str, Any]], pool: _RunnerPool,
                 cfg: Dict[str, Any], jsonl_path: str,
                 done_map: Dict[str, Dict[str, Any]]) -> str:
    n_steps = max(1, len(plan.get("steps") or []))
    pace_every = max(1, int(cfg.get("pace_every_queries", 30) or 30))

    pace_vids = max(1, pace_every // n_steps)
    pace_sleep = float(cfg.get("pace_sleep_s", 0.2))
    min_done = int(cfg.get("fail_breaker_min_done", 100) or 100)
    ratio = float(cfg.get("fail_breaker_ratio", 0.8) or 0.8)
    concurrency = max(1, int(cfg.get("concurrency", 1) or 1))
    todo = [t for t in panel if t["target_id"] not in done_map]
    if not todo:
        return "ok"
    logger.info(f"[True IV] Batch values: {len(todo)} vids pending ({len(done_map)} completed, "
                f"concurrency={concurrency}, throttling {pace_sleep}s per {pace_vids} vids)")

    _silence_exec_logs()
    try:
        return _batch_query_inner(plan, todo, pool, cfg, jsonl_path, done_map,
                                  pace_vids, pace_sleep, min_done, ratio,
                                  concurrency, t0=time.time())
    finally:
        _restore_exec_logs()


def _batch_query_inner(plan, todo, pool, cfg, jsonl_path, done_map,
                       pace_vids, pace_sleep, min_done, ratio,
                       concurrency, t0) -> str:
    done_n = 0
    fail_n = 0
    status = "ok"
    with open(jsonl_path, "a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="iv-vid") as ex:
            pending = set()
            it = iter(todo)
            def _fill() -> None:
                while len(pending) < concurrency:
                    try:
                        t = next(it)
                    except StopIteration:
                        return
                    pending.add(ex.submit(_query_one, plan, t, pool))
            _fill()
            stop = ""
            while pending and not stop:
                done_futs, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done_futs:
                    pending.discard(fut)
                    try:
                        rec = fut.result()
                    except _OutageAbort:
                        stop = "outage"
                        continue
                    done_n += 1
                    if rec.get("error"):
                        fail_n += 1
                    done_map[rec["target_id"]] = rec
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if done_n % 50 == 0:
                        fout.flush()
                        dt = time.time() - t0
                        logger.info(f"    [True IV] Progress {done_n}/{len(todo)}"
                                    f" (fail={fail_n}, avg={dt / done_n * 1000:.0f}ms/vid)")
                    if done_n % pace_vids == 0:
                        time.sleep(pace_sleep)
                    if not stop and done_n >= min_done and fail_n / done_n > ratio:
                        logger.warning(f"[True IV] Failure rate {fail_n}/{done_n} > {ratio:.0%}"
                                       f" (completed >={min_done}); circuit-breaking remaining queries for this feature")
                        stop = "breaker"
                if not stop:
                    _fill()
            if stop:
                for f in pending:
                    f.cancel()
                pending.clear()
            status = stop or "ok"
        if status == "breaker":
            n_stub = 0
            for t in todo:
                tid = t["target_id"]
                if tid in done_map:
                    continue
                stub = {"target_id": tid, "value": None, "tag": t["tag"],
                        "error": "fail_breaker_skip"}
                done_map[tid] = stub
                fout.write(json.dumps(stub, ensure_ascii=False) + "\n")
                n_stub += 1
            logger.warning(f"[True IV] Added {n_stub} fail_breaker_skip placeholder records")
        fout.flush()
    logger.info(f"[True IV] Batch completed: {done_n}/{len(todo)} queried (fail={fail_n}), "
                f"elapsed {time.time() - t0:.0f}s, status {status}")
    return status


_SPLITTER: Optional[AutobinSplit] = None


def _splitter_kwargs(cfg: Dict[str, Any]) -> dict:
    b = cfg.get("binning") or {}
    return {
        "num_bins_numeric": int(b.get("num_bins_numeric", 10)),
        "special_mass_ratio": float(b.get("special_mass_ratio", 0.5)),
        "min_numeric_bin_ratio": float(b.get("min_numeric_bin_ratio", 0.01)),
        "low_cardinality_threshold": int(b.get("low_cardinality_threshold", 10)),
        "cat_min_ratio": float(b.get("cat_min_ratio", 0.05)),
        "smoothing_factor": float(b.get("smoothing_factor", 0.5)),
    }


def _get_splitter(cfg: Dict[str, Any]) -> AutobinSplit:
    global _SPLITTER
    if cfg.get("evaluation_backend") == "fraudfeat":
        from ..execution.parquet_backend import fraudfeat_module
        cls = fraudfeat_module("autobin").AutobinSplit
        return cls(**_splitter_kwargs(cfg))
    if _SPLITTER is None:
        _SPLITTER = AutobinSplit(**_splitter_kwargs(cfg))
    return _SPLITTER


def _iv_from_records(records: List[Dict[str, Any]], splitter: AutobinSplit):
    rows = [{"value": r.get("value"), "label": 1 if r.get("tag") == "black" else 0}
            for r in records]
    if not rows:
        return None, {}
    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    metrics, _bins = splitter.split_evaluation(df, feature_cols=["value"], label_col="label")
    m = metrics.get("value")
    if m is None:
        return None, {}
    side = {"iv": m.iv, "n_bins": m.n_bins, "missing_rate": m.missing_rate,
            "min": m.min, "max": m.max, "mean": m.mean, "std": m.std}
    return float(m.iv), side


def _plan_key(plan: Dict[str, Any]) -> str:
    blob = json.dumps(plan, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()[:10]


def _safe_name(name: Any) -> str:
    s = re.sub(r'[\\/:*?"<>|\r\n\t\s]+', "_", str(name)).strip("._")
    return (s or "feature")[:80]


def _jsonl_path(cfg: Dict[str, Any], feature_name: str, key: str) -> str:
    configured = str(cfg.get("results_dir") or "runs/default/iv_values")
    use_run_env = bool(cfg.get("use_run_env", True))
    directory = os.environ.get("IV_RESULTS_DIR") if use_run_env else ""
    d = _resolve(directory or configured)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{_safe_name(feature_name)}_{key}.jsonl")


def _load_jsonl(path: str) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if not os.path.isfile(path):
        return done
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict) and rec.get("target_id"):
                    done[str(rec["target_id"])] = rec
    except Exception as e:
        logger.warning(f"[True IV] Cache read failed (ignored; rerunning batch): {path} ({e})")
        return {}
    return done


def _write_sidecar(jsonl_path: str, cand: Candidate, key: str,
                   panel: List[Dict[str, Any]], side: Dict[str, Any],
                   done_map: Dict[str, Dict[str, Any]]) -> None:
    valid = sum(1 for r in done_map.values() if r.get("value") is not None)
    doc = {
        "feature_name": cand.feature_name,
        "plan_key": key,
        "panel": {"black": sum(1 for p in panel if p["tag"] == "black"),
                  "white": sum(1 for p in panel if p["tag"] == "white")},
        "vid_total": len(done_map),
        "vid_valid_value": valid,
        **side,
    }
    try:
        with open(re.sub(r"\.jsonl$", ".metrics.json", jsonl_path), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[True IV] Sidecar write failed (main flow unaffected): {e}")


_OUTAGE_UNTIL = 0.0


def compute_real_iv(cand: Candidate, *, panel: Optional[List[Dict[str, Any]]] = None,
                    executor: Any = None, cfg: Optional[Dict[str, Any]] = None) -> Optional[float]:
    global _OUTAGE_UNTIL
    cfg = dict(cfg or load_iv_config())
    public_backend = getattr(executor, "backend", "") == "fraudfeat-parquet"
    if public_backend:
        cfg["evaluation_backend"] = "fraudfeat"
        cfg["auto_collect"] = False
        cfg.setdefault("pace_sleep_s", 0.0)
    cooldown = float(cfg.get("outage_cooldown_s", 600) or 600)
    if time.time() < _OUTAGE_UNTIL:
        logger.info(f"[True IV] Server-outage cooldown active; skipping batch (feature {cand.feature_name}, frozen)")
        return None

    plan = getattr(cand, "gremlin_plan", None)
    if not (isinstance(plan, dict) and plan.get("steps")):
        logger.warning(f"[True IV] {cand.feature_name} has no gremlin_plan steps; cannot collect batch values; freezing")
        return None

    plan = normalize_plan(plan)
    ok, verr = validate_plan(plan)
    if not ok:
        logger.warning(f"[True IV] {cand.feature_name} has an invalid plan structure ({verr}); freezing")
        return None

    if panel is None and public_backend:
        panel = list(getattr(executor, "offline_panel", None) or [])
    if panel is None:
        panel = _load_panel_cached(cfg)
        if panel:
            panel = _depth_filter_panel(panel, cfg)
    if not panel:
        logger.warning("[True IV] Evaluation panel missing/empty; prepare the configured fixed "
                       "sample panel before measured IV computation; candidate frozen")
        return None
    pool = _get_pool(executor)
    try:
        runner = pool.acquire()
    except Exception as e:
        logger.warning(f"[True IV] {e}; candidate frozen")
        return None
    try:
        fatal = _env_fatal(runner)
    finally:
        pool.release(runner)
    if fatal:
        _OUTAGE_UNTIL = time.time() + cooldown
        logger.warning(f"[True IV] Execution environment unavailable ({fatal[:120]}); cooldown {cooldown:.0f}s; candidate frozen")
        return None
    key = _plan_key(plan)
    if public_backend:
        panel_blob = json.dumps([(p.get("target_id"), p.get("tag")) for p in panel],
                                ensure_ascii=False, separators=(",", ":"))
        panel_key = hashlib.md5(panel_blob.encode("utf-8")).hexdigest()[:10]
        key = f"{key}_{panel_key}"
    jsonl_path = _jsonl_path(cfg, cand.feature_name, key)
    done_map = _load_jsonl(jsonl_path)
    if done_map:
        logger.info(f"[True IV] Resuming from checkpoint: {len(done_map)} existing records ({os.path.basename(jsonl_path)})")
    t_batch = time.time()
    status = _batch_query(plan, panel, pool, cfg, jsonl_path, done_map)
    if status == "outage":
        _OUTAGE_UNTIL = time.time() + cooldown
        logger.warning(f"[True IV] Server remains unavailable; aborting candidate with cooldown {cooldown:.0f}s"
                       f" (retaining {len(done_map)} written records; resume after recovery)")
        return None
    records = list(done_map.values())
    iv, side = _iv_from_records(records, _get_splitter(cfg))
    if iv is None:
        logger.warning(f"[True IV] {cand.feature_name} has no valid value records; freezing")
        return None
    _write_sidecar(jsonl_path, cand, key, panel, side, done_map)
    valid = sum(1 for r in records if r.get("value") is not None)
    if valid < 50:
        logger.warning(f"[True IV] {cand.feature_name} has only {valid}/{len(records)} valid values; "
                       f"IV has limited statistical significance (missing values still enter the missing bin)")
    pred = getattr(cand, "predicted_iv", None)
    gap_txt = f", predicted={pred:.3f}, delta={abs(iv - pred):.3f}" if pred is not None else ""
    logger.info(f"[True IV] {cand.feature_name}: IV={iv:.4f}{gap_txt}, "
                f"vid={len(records)} (valid {valid}), batch elapsed {time.time() - t_batch:.0f}s, "
                f"cache {os.path.basename(jsonl_path)}")
    return round(iv, 4)
