from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from ..core.schema import IVTier
from ..paths import DATA_DIR, MEMORY_DIR, PROJECT_ROOT
from ..utils import ensure_dir, get_logger, load_config


logger = get_logger("coldstart")
CLEANING_VERSION = "mr_v1"

_WORKSPACE = PROJECT_ROOT
_CLEANED_DIR = DATA_DIR / "coldstart" / "cleaned"
_REPORT_PATH = DATA_DIR / "coldstart" / "cleaning_report.json"
_MARKER_PATH = MEMORY_DIR / "coldstart_state.json"


_JF_HAS_RE = re.compile(r"has\(\s*['\"]join_field['\"]\s*,\s*(.*?)\)", re.DOTALL)
_JF_STR_RE = re.compile(r"['\"]([^'\"]+)['\"]")
_ANCHOR_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SKILL_SYSTEM = """Summarize a cluster of structurally similar measured features into one reusable hypothesis rule.
Return English JSON with anchor (snake_case), structure (transferable chain and signals), feature_form (count/flag/ratio/amount and measured object), discrimination (metric and comparison conditions), why (structural risk mechanism), and boundary (when not applicable).
Do not bind the rule to concrete feature names, node IDs, absolute amounts, IV values, or files. Return {"skip":true} when no transferable commonality exists. Return JSON only."""


def _load_manifest() -> dict:
    try:
        return load_config("coldstart.yaml") or {}
    except Exception as e:
        logger.warning(f"[Cold start] Failed to load coldstart.yaml ({e}); treating as disabled")
        return {}


def run_for(*, account_type: str, eval_memory, skill_memory, llm=None,
            cli_enabled: bool = False, cli_reset: bool = False,
            persist_ok: bool = True, cfg: dict | None = None) -> dict:
    cfg = cfg if cfg is not None else _load_manifest()
    enabled = bool(cli_enabled) or bool(cfg.get("enabled"))
    reset = bool(cli_reset) or bool(cfg.get("reset"))
    if not (enabled or reset):
        return {}
    if not persist_ok:
        logger.warning("[Cold start] Memory persistence is disabled (--no-eval-memory/--no-skill-memory, etc.); "
                       "injections cannot survive across runs; skipping to avoid wasted LLM calls")
        return {}
    types = ["individual", "merchant"] if account_type == "all" else [str(account_type)]
    out: dict = {}
    for t in types:
        if reset:
            reset_for(t, eval_memory=eval_memory, skill_memory=skill_memory)
        if enabled:
            out[t] = _run_one(t, cfg, eval_memory=eval_memory,
                              skill_memory=skill_memory, llm=llm)
    return out


def reset_for(account_type: str, *, eval_memory, skill_memory) -> dict:
    marker = _load_marker()
    rec = marker.pop(account_type, None)
    out = {"eval_removed": 0, "skills_removed": 0}
    if not rec:
        logger.info(f"[Cold start] {account_type}: no marker record; nothing to clear")
        return out
    out["eval_removed"] = eval_memory.remove(set(rec.get("eval_doc_names") or []))
    out["skills_removed"] = skill_memory.remove(set(rec.get("skill_ids") or []))
    _save_marker(marker)
    logger.info(f"[Cold start] {account_type}: artifacts removed (evaluator memory -{out['eval_removed']}, "
                f"skills -{out['skills_removed']}); marker cleared; cleaned artifacts retained")
    return out


def _purge_rec(rec: dict, eval_memory, skill_memory) -> None:
    n_eval = eval_memory.remove(set(rec.get("eval_doc_names") or []))
    n_sk = skill_memory.remove(set(rec.get("skill_ids") or []))
    if n_eval or n_sk:
        logger.info(f"[Cold start] Removed old artifacts before rebuilding: evaluator -{n_eval}, skills -{n_sk}")


def dry_run(account_type: str = "individual", cfg: dict | None = None) -> dict:
    cfg = cfg if cfg is not None else _load_manifest()
    inject_cfg = cfg.get("inject") or {}
    min_cluster = int(inject_cfg.get("skill_min_cluster", 3))
    kept, reports = _clean_sources(cfg, account_type)
    clusters = _cluster(kept) if kept else {}
    _write_report(account_type, reports, kept, clusters, min_cluster)
    big = {s: m for s, m in clusters.items() if len(m) >= min_cluster}
    logger.info(f"[Cold start][Dry run] {account_type}: {len(kept)} entries after cleaning, "
                f"{len(clusters)} clusters ({len(big)} with >={min_cluster} entries); no injection")
    return {"kept": len(kept), "reports": reports,
            "clusters_total": len(clusters), "clusters_ge_min": len(big)}


def _run_one(account_type: str, cfg: dict, *, eval_memory, skill_memory, llm) -> dict:
    marker = _load_marker()
    rec = marker.get(account_type)
    if rec and rec.get("cleaning_version") == CLEANING_VERSION:
        ok, why = _verify_marker(rec, eval_memory, skill_memory)
        if ok:
            changed = _sources_changed(rec, cfg, account_type)
            if changed:
                logger.warning(f"[Cold start] {account_type}: source files changed ({changed}); "
                               "use --cold-start-reset to rebuild from new sources (no automatic rebuild; "
                               "changing data is an explicit decision)")
            logger.info(f"[Cold start] {account_type}: completed at {rec.get('done_at')} and verified; "
                        "skipping repeated initialization across batches; use --cold-start-reset to rebuild")
            return {"status": "skipped"}
        logger.warning(f"[Cold start] {account_type}: marker verification failed ({why}); "
                       "memory store may have been cleared; rebuilding automatically")
        _purge_rec(rec, eval_memory, skill_memory)
    elif rec:
        logger.warning(f"[Cold start] {account_type}: cleaning-rule version changed"
                       f" ({rec.get('cleaning_version')} -> {CLEANING_VERSION}); rebuilding automatically")
        _purge_rec(rec, eval_memory, skill_memory)
    inject_cfg = cfg.get("inject") or {}
    cap = int(inject_cfg.get("eval_cap", 150))
    min_cluster = int(inject_cfg.get("skill_min_cluster", 3))
    concurrency = int(inject_cfg.get("concurrency", 5))
    kept, reports = _clean_sources(cfg, account_type)
    if not kept:
        logger.warning(f"[Cold start] {account_type}: no usable entries after cleaning; aborting without writing a marker")
        return {"status": "empty"}
    doc_names = _inject_eval(eval_memory, kept, cap,
                             uniform=bool(inject_cfg.get("uniform_eval")))
    clusters = _cluster(kept)
    skill_ids = _inject_skills(skill_memory, llm, clusters, min_cluster,
                               concurrency=concurrency)

    _write_report(account_type, reports, kept, clusters, min_cluster)
    marker[account_type] = {
        "done_at": datetime.now().isoformat(timespec="seconds"),
        "cleaning_version": CLEANING_VERSION,
        "sources": [{"path": r["path"], "sha1": r["sha1"],
                     "raw": r["raw"], "kept": r["kept"]} for r in reports],
        "eval_doc_names": doc_names,
        "eval_injected": len(doc_names),
        "skill_ids": skill_ids,
        "skills_upserted": len(skill_ids),
    }
    _save_marker(marker)
    raw_total = sum(r.get("raw", 0) for r in reports)
    logger.info(f"[Cold start] {account_type} completed: cleaned {raw_total}->{len(kept)}"
                f" (dropped {raw_total - len(kept)}) | injected {len(doc_names)} evaluator entries"
                f" | summarized {len(skill_ids)} skills | marker written to {_MARKER_PATH}")
    return {"status": "done", "kept": len(kept),
            "eval_injected": len(doc_names), "skills": len(skill_ids)}


def _clean_sources(cfg: dict, account_type: str) -> tuple[list[dict], list[dict]]:
    cleaning = cfg.get("cleaning") or {}
    mr_drop = float(cleaning.get("mr_drop", 0.99))
    mr_soft = float(cleaning.get("mr_soft", 0.95))
    soft_iv_min = float(cleaning.get("soft_iv_min", 0.5))
    kept_all: list[dict] = []
    reports: list[dict] = []
    for src in cfg.get("sources") or []:
        suit = str(src.get("suitable") or "")
        if suit != account_type:
            continue
        path = _WORKSPACE / str(src.get("path") or "")
        if not path.exists():
            logger.warning(f"[Cold start] Source file does not exist; skipping: {path}")
            continue
        items = json.loads(path.read_text(encoding="utf-8"))
        sha1 = hashlib.sha1(path.read_bytes()).hexdigest()
        kept, rep = _clean_items(items, mr_drop=mr_drop, mr_soft=mr_soft,
                                 soft_iv_min=soft_iv_min)
        mismatch = [k for k in kept if str(k.get("suitable") or "") != account_type]
        if mismatch:
            logger.warning(f"[Cold start] {path.name}: skipped {len(mismatch)} entries whose suitable field "
                           f"does not match the manifest label ({account_type})")
            kept = [k for k in kept if str(k.get("suitable") or "") == account_type]
        rep.update({"path": str(src.get("path")), "suitable": suit, "sha1": sha1,
                    "raw": len(items), "kept": len(kept),
                    "suitable_mismatch": len(mismatch)})
        reports.append(rep)
        for k in kept:
            k["_source"] = path.stem
        kept_all.extend(kept)
        cleaned_path = _write_cleaned(path.stem, kept)
        logger.info(f"[Cold start] Cleaned {path.name}: {len(items)}->{len(kept)}"
                    f" (basic-validation drops {rep['drop_basic']}, missing-rate drops {rep['drop_missing']}, "
                    f"soft-range exemptions {rep['soft_exempt']}) -> {cleaned_path}")
    return kept_all, reports


def _clean_items(items: list[dict], *, mr_drop: float, mr_soft: float,
                 soft_iv_min: float) -> tuple[list[dict], dict]:
    kept: list[dict] = []
    drop_basic = drop_missing = soft_exempt = 0
    for it in items:
        steps = (it.get("gremlin_plan") or {}).get("steps") or []
        iv = it.get("iv")
        logic = str(it.get("logic") or "").strip()
        if not steps or iv is None or not logic:
            drop_basic += 1
            continue
        mr = float(it.get("missing_rate") or 0.0)
        ivf = float(iv)
        if mr >= mr_drop:
            drop_missing += 1
            continue
        if mr_soft < mr < mr_drop:
            if ivf < soft_iv_min:
                drop_missing += 1
                continue
            soft_exempt += 1
        it["_tier"] = IVTier.from_value(ivf).value
        it["_signature"] = list(_struct_signature(it))
        kept.append(it)
    return kept, {"drop_basic": drop_basic, "drop_missing": drop_missing,
                  "soft_exempt": soft_exempt}


def _struct_signature(item: dict) -> tuple:
    plan = item.get("gremlin_plan") or {}
    fams: list[str] = []
    for s in plan.get("steps") or []:
        g = str(s.get("gremlin") or "")
        jfs: list[str] = []
        for seg in _JF_HAS_RE.findall(g):
            jfs.extend(x for x in _JF_STR_RE.findall(seg) if x != "join_field")
        fams.append("+".join(sorted({jf.split("|", 1)[1] if "|" in jf else jf
                                     for jf in jfs})) or "?")
    ftype = str((plan.get("final") or item.get("final") or {}).get("type") or "?")
    return (ftype, tuple(fams))


def _to_seed(item: dict) -> tuple[str, float, str]:
    name = str(item.get("feature_name") or "").strip()
    plan = item.get("gremlin_plan") or {}
    steps = plan.get("steps") or []
    structure = f"hops={len(steps)}"
    notes: list[str] = []
    for s in steps:
        d = str(s.get("desc") or "").strip()
        if d and d not in notes:
            notes.append(d)
    if notes:
        structure += f", chain[{' → '.join(n[:48] for n in notes[:3])}]"
    risk = (str(plan.get("feature_understanding") or "").strip()
            or str(item.get("logic") or "")[:120])
    risk = risk.replace("\n", " ")
    desc = f"Feature: {name}\nStructure: {structure}\nRisk rationale: {risk}"
    return desc, float(item.get("iv")), name


def _stratified_sample(kept: list[dict], cap: int) -> list[dict]:
    lows_mids = [k for k in kept if k.get("_tier") != "high"]
    highs = [k for k in kept if k.get("_tier") == "high"]
    best: dict[tuple, dict] = {}
    for k in highs:
        sig = tuple(k.get("_signature") or ())
        if sig not in best or float(k.get("iv") or 0.0) > float(best[sig].get("iv") or 0.0):
            best[sig] = k
    uniq_highs = sorted(best.values(), key=lambda x: float(x.get("iv") or 0.0),
                        reverse=True)
    quota = max(0, cap - len(lows_mids))
    return lows_mids + uniq_highs[:quota]


def _inject_eval(eval_memory, kept: list[dict], cap: int,
                 uniform: bool = False) -> list[str]:
    picked = kept[:cap] if uniform else _stratified_sample(kept, cap)
    seeds = [_to_seed(k) for k in picked]
    eval_memory.bootstrap(seeds)
    eval_memory.save()
    n_lm = sum(1 for k in picked if k.get("_tier") != "high")
    logger.info(f"[Cold start] Channel 1 evaluator memory: stratified injection of {len(picked)} entries"
                f" (all low/mid {n_lm} + signature-deduplicated high samples {len(picked) - n_lm}, "
                f"cap={cap}; sent to RAG with warm_items during the offline phase)")
    return [name for _, _, name in seeds]


def _cluster(kept: list[dict]) -> dict[tuple, list[dict]]:
    clusters: dict[tuple, list[dict]] = {}
    for k in kept:
        sig = tuple(k.get("_signature") or ())
        clusters.setdefault(sig, []).append(k)
    return clusters


def _inject_skills(skill_memory, llm, clusters: dict, min_cluster: int,
                   concurrency: int = 5) -> list[str]:
    big = [(sig, ms) for sig, ms in clusters.items() if len(ms) >= min_cluster]
    if not big:
        logger.info(f"[Cold start] Channel 3 skill library: no clusters with >={min_cluster} entries; skipping with zero LLM calls")
        return []
    if llm is None:
        logger.warning("[Cold start] Channel 3 skill library: no LLM client; skipping skill summarization")
        return []
    from ..parallel import run_parallel
    got_list = run_parallel(
        big, lambda item, i: _summarize_cluster(llm, item[0], item[1]),
        max_workers=int(concurrency), stage=f"Cold-start skill summarization: {len(big)} clusters")
    ids: list[str] = []
    for (sig, members), got in zip(big, got_list):
        if not got:
            continue
        anchor, fields = got
        skill_memory.add_rule("hypothesize", anchor, fields, source="cold start")
        ids.append(anchor)
    if ids:
        skill_memory.save()
    logger.info(f"[Cold start] Channel 3 skill library: {len(clusters)} clusters ({len(big)} with >={min_cluster} entries), "
                f"{len(big)} LLM calls at concurrency {int(concurrency)}, summarized {len(ids)} skills"
                f" (five-field rule entries written to feature_hypothesize; merged during first-episode revision)")
    return ids


def _summarize_cluster(llm, sig: tuple, members: list[dict]) -> tuple | None:
    resp = llm.complete_json(_SKILL_SYSTEM, _cluster_prompt(sig, members),
                             temperature=0.3)
    if not isinstance(resp, dict) or resp.get("skip"):
        return None
    anchor = str(resp.get("anchor") or "").strip()
    if not _ANCHOR_RE.match(anchor):
        logger.warning(f"[Cold start] Invalid cluster skill summary (anchor={anchor!r}); skipping cluster")
        return None
    fields = {
        "structure": str(resp.get("structure") or "").strip(),
        "feature_form": str(resp.get("feature_form") or "").strip(),
        "discrimination": str(resp.get("discrimination") or "").strip(),
        "why": str(resp.get("why") or "").strip(),
        "boundary": str(resp.get("boundary") or "").strip(),
    }
    if not fields["structure"] or not fields["feature_form"]:
        logger.warning(f"[Cold start] Required cluster summary fields missing (anchor={anchor!r}); skipping cluster")
        return None
    return anchor, fields


def _cluster_prompt(sig: tuple, members: list[dict]) -> str:
    ftype, fams = sig
    ivs = sorted(float(m.get("iv") or 0.0) for m in members)
    mrs = sorted(float(m.get("missing_rate") or 0.0) for m in members)
    lines = []
    for m in members[:8]:
        und = str((m.get("gremlin_plan") or {}).get("feature_understanding") or "")
        und = und.replace("\n", " ")[:100]
        lines.append(f"- {m.get('feature_name')} | IV={float(m.get('iv') or 0.0):.3f}"
                     f" | missing_rate={float(m.get('missing_rate') or 0.0):.3f} | {und}")
    return (f"[STRUCTURAL CLUSTER] reduction={ftype} | relations={' -> '.join(fams) or 'none'}"
            f" | size={len(members)}\n"
            f"IV range: {ivs[0]:.3f} to {ivs[-1]:.3f}; "
            f"missing-rate range: {mrs[0]:.3f} to {mrs[-1]:.3f}\n"
            f"Examples ({len(lines)}/{len(members)}):\n" + "\n".join(lines)
            + "\n\nReturn one transferable rule following the system schema, or {\"skip\":true}.")


def protected_skill_slugs() -> set:
    marker = _load_marker()
    out: set = set()
    for rec in marker.values():
        if isinstance(rec, dict):
            out.update(str(s) for s in (rec.get("skill_ids") or []))
    return out


def protected_eval_doc_names() -> set:
    marker = _load_marker()
    out: set = set()
    for rec in marker.values():
        if isinstance(rec, dict):
            out.update(str(s) for s in (rec.get("eval_doc_names") or []))
    return out


def _load_marker() -> dict:
    if not _MARKER_PATH.exists():
        return {}
    try:
        return json.loads(_MARKER_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"[Cold start] Marker read failed ({e}); treating as no record: {_MARKER_PATH}")
        return {}


def _save_marker(marker: dict) -> None:
    ensure_dir(_MARKER_PATH.parent)
    _MARKER_PATH.write_text(json.dumps(marker, ensure_ascii=False, indent=2),
                            encoding="utf-8")


def _verify_marker(rec: dict, eval_memory, skill_memory) -> tuple[bool, str]:
    n_eval = int(rec.get("eval_injected") or 0)
    if n_eval and len(eval_memory.memory) == 0:
        return False, "evaluator memory is empty"
    existing = {m.doc_name for m in eval_memory.memory}
    missing = [n for n in (rec.get("eval_doc_names") or [])[:3] if n not in existing]
    if missing and len(eval_memory.memory) < n_eval:
        return False, (f"evaluator memory lacks doc_name={missing[0]} and size "
                       f"{len(eval_memory.memory)} < injected count {n_eval}")
    n_sk = int(rec.get("skills_upserted") or 0)
    _cc = getattr(skill_memory, "card_count", None)
    n_cards = _cc("hypothesize") if callable(_cc) else len(skill_memory.skills)
    if n_sk and n_cards == 0:
        return False, "skill library is empty"
    return True, ""


def _sources_changed(rec: dict, cfg: dict, account_type: str) -> str:
    old = {s.get("path"): s.get("sha1") for s in rec.get("sources") or []}
    for src in cfg.get("sources") or []:
        if str(src.get("suitable") or "") != account_type:
            continue
        p = _WORKSPACE / str(src.get("path") or "")
        if not p.exists():
            return f"{src.get('path')} (file missing)"
        if old.get(str(src.get("path"))) != hashlib.sha1(p.read_bytes()).hexdigest():
            return str(src.get("path"))
    return ""


def _write_cleaned(stem: str, kept: list[dict]) -> Path:
    ensure_dir(_CLEANED_DIR)
    p = _CLEANED_DIR / f"{stem}.json"
    p.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def _write_report(account_type: str, reports: list[dict], kept: list[dict],
                  clusters: dict, min_cluster: int) -> None:
    from collections import Counter
    tier_dist = Counter(k.get("_tier") for k in kept)
    clist = []
    for sig, ms in clusters.items():
        ivs = [float(m.get("iv") or 0.0) for m in ms]
        clist.append({"signature": list(sig), "size": len(ms),
                      "iv_min": round(min(ivs), 4), "iv_max": round(max(ivs), 4),
                      "skill_summarized": len(ms) >= min_cluster})
    clist.sort(key=lambda c: -c["size"])
    doc: dict = {}
    if _REPORT_PATH.exists():
        try:
            doc = json.loads(_REPORT_PATH.read_text(encoding="utf-8"))
        except Exception:
            doc = {}
    doc[account_type] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cleaning_version": CLEANING_VERSION,
        "sources": reports,
        "kept_total": len(kept),
        "tier_dist": dict(tier_dist),
        "clusters_total": len(clusters),
        "clusters_ge_min": sum(1 for c in clist if c["skill_summarized"]),
        "cluster_top": clist[:20],
    }
    ensure_dir(_REPORT_PATH.parent)
    _REPORT_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                            encoding="utf-8")
