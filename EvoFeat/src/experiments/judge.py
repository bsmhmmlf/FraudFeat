from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

from . import judge_domain
from ._common import emit, rank_by_iv, read_pool, write_txt
from system.execution.parquet_backend import fraudfeat_module


JUDGES = [m.strip() for m in os.environ.get(
    "JUDGE_MODELS",
    "deepseek-v4-pro,gpt-6.1-sol,claude-opus-5.5").split(",") if m.strip()]
JUDGE_API_URL = os.environ.get("JUDGE_API_URL", "<YOUR_JUDGE_API_URL>")
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "<YOUR_JUDGE_API_KEY>")
JUDGE_TIMEOUT = (5.0, 300.0)


def _protocol():
    return fraudfeat_module("interpretability")


RUBRIC_PROMPT = _protocol().RUBRIC_PROMPT
_system_cache: str | None = None


def system_prompt() -> str:
    global _system_cache
    if _system_cache is None:
        _system_cache = _protocol().system_prompt(judge_domain.domain_brief())
    return _system_cache


RUBRIC_SHA1 = _protocol().rubric_sha1(judge_domain.domain_brief())


BASELINE_PREFIXES = ("stat", "comm", "manual")


def _parse_judge_json(content: str) -> dict | None:
    return _protocol()._parse_judge_json(content)


class JudgeAdapter:
    def __init__(self, judge: str, api_url: str | None = None,
                 api_key: str | None = None, max_retries: int = 2,
                 timeout=None):
        if judge not in JUDGES:
            raise ValueError(f"Unknown judge {judge} (allowed: {JUDGES})")
        self.judge = judge
        self.api_url = api_url or JUDGE_API_URL
        self.api_key = api_key or JUDGE_API_KEY
        self.max_retries = max_retries
        self.timeout = timeout or JUDGE_TIMEOUT

    def _prompt(self, card: dict) -> str:
        name = str(card.get("feature_name") or "").strip()
        logic = str(card.get("logic") or "None").strip()
        expl = str(card.get("explanation") or "None").strip()
        return f"Feature name: {name}\nComputation logic: {logic}\nBusiness explanation: {expl}"

    def score(self, feature_card: dict) -> dict:
        import requests
        payload = {"model": self.judge,
                   "messages": [{"role": "system", "content": system_prompt()},
                                {"role": "user", "content": self._prompt(feature_card)}],
                   "stream": False, "temperature": 0, "max_tokens": 256}
        last_err = "Request not started"
        for attempt in range(1 + self.max_retries):
            try:
                r = requests.post(
                    self.api_url,
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    json=payload, timeout=self.timeout)
                if r.status_code != 200:
                    err = r.text or ""
                    for k in ("temperature", "max_tokens"):
                        if k in payload and k in err.lower():
                            payload.pop(k, None)
                    last_err = f"HTTP {r.status_code}: {err[:100]}"
                else:
                    try:
                        content = r.json()["choices"][0]["message"]["content"]
                    except Exception as e:
                        last_err = f"Response parsing failed: {e}"
                        continue
                    parsed = _parse_judge_json(content)
                    if parsed:
                        return parsed
                    last_err = f"Output is not JSON with a score from 1 to 5: {content[:100]}"
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:100]}"
            if attempt < self.max_retries:
                time.sleep(1.5 * (attempt + 1))
        return {"score": None, "rationale": last_err}

    def probe(self) -> bool:
        res = self.score({"feature_name": "connectivity_probe", "logic": "-", "explanation": "-"})
        if res.get("score") is not None:
            return True
        print(f"[judge][probe] {self.judge} unavailable: {res.get('rationale', '')}")
        return False


def score_run(run_dir: str | Path, judges: list[str] | None = None,
              limit: int = 0, workers: int = 4, force: bool = False) -> dict:
    run_dir = Path(run_dir)
    req_path = run_dir / "judge_requests.jsonl"
    if not req_path.is_file():
        return {"scored": 0, "skipped": 0, "failed": 0}
    reqs = _read_jsonl(req_path)
    if limit > 0:
        reqs = reqs[:limit]
    scores_path = run_dir / "scores.jsonl"
    prev_rows = _read_jsonl(scores_path)
    done = set() if force else {r.get("request_id") for r in prev_rows}
    todo = [r for r in reqs if r.get("request_id") and r["request_id"] not in done]
    judges = judges or JUDGES
    adapters = {j: JudgeAdapter(j) for j in judges}
    stats = {"scored": 0, "skipped": len(reqs) - len(todo), "failed": 0}

    def _one(r: dict) -> dict | None:
        if r.get("judge") not in adapters:
            stats["failed"] += 1
            return None
        res = adapters[r["judge"]].score(
            {"feature_name": r.get("feature_name"),
             "logic": r.get("logic"),
             "explanation": r.get("explanation")})
        if res.get("score") is None:
            stats["failed"] += 1
            print(f"[judge][failed] {r.get('judge')} '{r.get('feature_name')}': "
                  f"{res.get('rationale', '')[:80]}")
            return None
        return {"request_id": r.get("request_id"), "run_id": r.get("run_id"),
                "feature_name": r.get("feature_name"),
                "pool_index": r.get("pool_index"),
                "judge": r.get("judge"),
                "score": res["score"], "rationale": res["rationale"],
                "repeat": int(r.get("repeat") or 0), "rubric_sha1": RUBRIC_SHA1}

    rows: list[dict] = []
    if workers <= 1:
        for r in todo:
            row = _one(r)
            if row:
                rows.append(row)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for row in ex.map(_one, todo):
                if row:
                    rows.append(row)
    if rows:
        todo_ids = {r.get("request_id") for r in todo}
        with open(scores_path, "w" if force else "a", encoding="utf-8") as fp:
            if force:
                for r in prev_rows:
                    if r.get("request_id") not in todo_ids:
                        fp.write(json.dumps(r, ensure_ascii=False) + "\n")
            for row in rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats["scored"] = len(rows)
    print(f"[judge] {run_dir.name}: newly scored {stats['scored']} / "
          f"skipped (already scored) {stats['skipped']} / failed {stats['failed']} → {scores_path}")
    return stats


def method_of(run_id: str) -> str:
    return str(run_id).split("__", 1)[0]


def explanation_of(f: dict, run_id: str) -> tuple[str, str]:
    m = method_of(run_id)
    if m.startswith(BASELINE_PREFIXES):
        return str(f.get("rationale") or ""), "rationale"
    return str(f.get("business_explanation") or ""), "business_explanation"


def feature_card(f: dict, run_id: str = "") -> dict:
    expl, src = explanation_of(f, run_id)
    normalized = dict(f)
    normalized.pop("display_name", None)
    normalized.update({
        "explanation": expl,
        "explanation_source": src,
    })
    return _protocol().feature_card(normalized)


def build_requests(run_id: str, run_dir: str | Path, repeat: int = 1,
                   stability_features: int = 0, stability_repeat: int = 3,
                   seed: int = 20260817, pool_limit: int = 0) -> list[dict]:
    pool = read_pool(run_dir, limit=pool_limit)
    rep_idx: set = set()
    if stability_features > 0 and pool:
        rep_idx = set(random.Random(seed).sample(
            list(range(len(pool))), min(stability_features, len(pool))))
    reqs: list[dict] = []
    for idx, f in enumerate(pool):
        fn = str(f.get("feature_name") or "")
        if not fn:
            continue
        card = feature_card(f, run_id)
        raw_index = f.get("pool_index")
        pool_index = raw_index if isinstance(raw_index, int) and not isinstance(raw_index, bool) else idx
        card_sha1 = hashlib.sha1(json.dumps(
            card, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        n = max(1, stability_repeat) if idx in rep_idx else max(1, repeat)
        for j in JUDGES:
            for rep in range(n):
                reqs.append({
                    "request_id": f"{run_id}|{pool_index}|{fn}|{j}|{rep}|{card_sha1}",
                    "run_id": run_id, "judge": j, "repeat": rep,
                    "pool_index": pool_index,
                    "entry_key": f"{fn}#{pool_index}",
                    "card_sha1": card_sha1,
                    "rubric_sha1": RUBRIC_SHA1,
                    "true_tier": f.get("true_tier"),
                    "true_iv": f.get("true_iv"),
                    **card})
    return reqs


def write_request_jsonl(run_dir: str | Path, run_id: str = "", **kw) -> Path:
    run_dir = Path(run_dir)
    run_id = run_id or run_dir.name
    path = run_dir / "judge_requests.jsonl"
    with open(path, "w", encoding="utf-8") as fp:
        for r in build_requests(run_id, run_dir, **kw):
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.is_file():
        return out
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def _norm_score_row(r: dict, by_req: dict, run_of: dict) -> dict | None:
    rid = str(r.get("request_id") or "")
    req = by_req.get(rid) if rid else None
    run_id = str(r.get("run_id") or (req or {}).get("run_id")
                 or run_of.get(rid, ""))
    fn = r.get("feature_name") or (req or {}).get("feature_name")
    j = r.get("judge") or (req or {}).get("judge")
    score = r.get("score")
    if not run_id or not fn or not j or not isinstance(score, (int, float)):
        return None
    return {"run_id": run_id, "feature_name": fn, "judge": j,
            "score": int(score), "repeat": int(r.get("repeat") or (req or {}).get("repeat") or 0),
            "rationale": str(r.get("rationale") or ""),
            "rubric_sha1": str(r.get("rubric_sha1") or RUBRIC_SHA1)}


def ingest_scores(src: str | Path, runs_root: str | Path = "runs",
                  force: bool = True) -> dict:
    runs_root = Path(runs_root)
    src = Path(src)
    if src.is_dir():
        files = [p for p in sorted(src.rglob("*.jsonl")) if p.name != "scores.jsonl"]
    else:
        files = [src]
    by_req: dict[str, dict] = {}
    for p in sorted(runs_root.glob("*/judge_requests.jsonl")):
        for r in _read_jsonl(p):
            if r.get("request_id"):
                by_req[str(r["request_id"])] = r
    by_run: dict[str, list[dict]] = {}
    bad = 0
    for p in files:
        if not p.is_file():
            print(f"[judge][warning] Score file does not exist: {p}")
            continue
        for r in _read_jsonl(p):
            row = _norm_score_row(r, by_req, {})
            if row is None:
                bad += 1
                continue
            by_run.setdefault(row["run_id"], []).append(row)
    known = {p.name for p in runs_root.iterdir() if p.is_dir()}
    unknown = sorted(set(by_run) - known)
    if unknown:
        print(f"[judge][warning] Score run_ids are not under {runs_root}/ (not written back): {unknown}")
    written = {}
    for run_id, rows in sorted(by_run.items()):
        d = runs_root / run_id
        if not d.is_dir():
            continue
        out = d / "scores.jsonl"
        if out.exists() and not force:
            print(f"[judge] {run_id}: scores.jsonl exists; skipping (--force to overwrite)")
            continue
        with open(out, "w", encoding="utf-8") as fp:
            for r in rows:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
        written[run_id] = len(rows)
        print(f"[judge] {run_id}: wrote back {len(rows)} lines → {out}")
    print(f"[judge] Ingestion complete: {len(written)} runs, {sum(written.values())} lines"
          + (f", discarded {bad} unrecognized lines" if bad else ""))
    return written


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    if n < 3 or len(b) != n:
        return None
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return round(num / (da * db), 4) if da and db else None


def kendalls_w(judge_rows: list[list[float]]) -> float | None:
    m = len(judge_rows)
    if m < 2:
        return None
    n = len(judge_rows[0])
    if n < 2 or any(len(r) != n for r in judge_rows):
        return None
    rank_rows = [_ranks(r) for r in judge_rows]
    rank_sums = [sum(rank_rows[j][i] for j in range(m)) for i in range(n)]
    mean_rank = m * (n + 1) / 2
    s = sum((r - mean_rank) ** 2 for r in rank_sums)
    tie_total = 0
    for row in judge_rows:
        counts = {}
        for value in row:
            counts[value] = counts.get(value, 0) + 1
        tie_total += sum(count ** 3 - count for count in counts.values() if count > 1)
    denominator = m * m * (n ** 3 - n) - m * tie_total
    if denominator <= 0:
        return None
    return round(12 * s / denominator, 4)


def _score_key(rec: dict) -> str:
    fn = str(rec.get("feature_name") or "")
    pi = rec.get("pool_index")
    return f"{fn}#{pi}" if isinstance(pi, int) else fn


def load_scores(run_dir: str | Path) -> dict[str, dict[str, list[int]]]:
    path = Path(run_dir) / "scores.jsonl"
    scores: dict[str, dict[str, list[int]]] = {}
    if not path.is_file():
        return scores
    for r in _read_jsonl(path):
        fn, j = r.get("feature_name"), r.get("judge")
        if fn and j and isinstance(r.get("score"), (int, float)):
            scores.setdefault(_score_key(r), {}).setdefault(str(j), []).append(int(r["score"]))
    return scores


TOPK_KS = (10, 30, 50, 100)


def _entry_score_key(f: dict, scores: dict) -> str | None:
    k = _score_key(f)
    if k in scores:
        return k
    fn = str(f.get("feature_name") or "")
    if fn in scores:
        return fn
    for sk in scores:
        if sk.split("#")[0] == fn:
            return sk
    return None


def topk_scores(run_dir: str | Path, pool_limit: int = 0,
                ks: tuple = TOPK_KS) -> dict[int, dict]:
    scores = load_scores(run_dir)
    if not scores:
        return {}
    feat_mean: dict[str, float] = {}
    for key, per in scores.items():
        vals = [per[j][0] for j in JUDGES if per.get(j)]
        if vals:
            feat_mean[key] = sum(vals) / len(vals)
    ranked = rank_by_iv(read_pool(run_dir, limit=pool_limit), run_dir)
    out: dict[int, dict] = {}
    for k in ks:
        vals = []
        for f in ranked[:k]:
            key = _entry_score_key(f, scores)
            if key and key in feat_mean:
                vals.append(feat_mean[key])
        if vals:
            out[k] = {"value": round(sum(vals) / len(vals), 4), "n": len(vals)}
    return out


def run_report(run_dir: str | Path, stability_sample: int = 20,
               seed: int = 20260817, pool_limit: int = 0) -> list[str]:
    run_dir = Path(run_dir)
    scores = load_scores(run_dir)
    pool = read_pool(run_dir, limit=pool_limit)
    n_pool = len(pool)
    if not scores:
        return [emit("judge_report", error="scores.jsonl is missing or empty",
                     pool=n_pool, rubric=RUBRIC_SHA1)]
    if pool_limit:
        scored_entries = [k for k in (_entry_score_key(f, scores) for f in pool) if k]
        feats = sorted(set(scored_entries))
        if not feats:
            return [emit("judge_report",
                         error="No scores in the analysis set (pool_limit and scores.jsonl disagree)",
                         pool=n_pool, rubric=RUBRIC_SHA1)]
        coverage = round(len(scored_entries) / n_pool, 4) if n_pool else None
    else:
        feats = sorted(scores)
        coverage = round(len(feats) / n_pool, 4) if n_pool else None
    means: dict[str, float] = {}
    for j in JUDGES:
        vals = [scores[f][j][0] for f in feats if scores[f].get(j)]
        if vals:
            means[j] = round(sum(vals) / len(vals), 4)
    full = [f for f in feats if all(scores[f].get(j) for j in JUDGES)]
    mat = [[scores[f][j][0] for f in full] for j in JUDGES]
    kendall = kendalls_w(mat)
    rng = random.Random(seed)
    repeated = [(f, j) for f in feats for j in JUDGES if len(scores[f].get(j, [])) > 1]
    sample = rng.sample(repeated, min(stability_sample, len(repeated)))
    variances = []
    for f, j in sample:
        vs = scores[f][j]
        m = sum(vs) / len(vs)
        variances.append(sum((v - m) ** 2 for v in vs) / len(vs))
    stability = (round(sum(variances) / len(variances), 4) if variances else None)
    overall = (round(sum(means.values()) / len(means), 4)
               if len(means) == len(JUDGES) else
               (round(sum(means.values()) / len(means), 4) if means else None))

    lines = [emit("judge_score", value=overall),
             emit("judge_w", value=kendall),
             emit("judge_coverage_rate", value=coverage)]
    if stability is not None:
        lines.append(emit("judge_stability", value=stability))
    for k, info in sorted(topk_scores(run_dir, pool_limit).items()):
        lines.append(emit("topk_judge_score", k=k,
                          value=info["value"], n=info["n"]))
    lines.append(emit("judge_rubric", sha1=RUBRIC_SHA1, judges=len(JUDGES),
                      domain=judge_domain.domain_sha1()))
    lines.append(emit("judge_coverage", scored=len(feats), pool=n_pool,
                      rate=coverage, rubric=RUBRIC_SHA1))
    for j, v in means.items():
        n_j = sum(1 for f in feats if scores[f].get(j))
        lines.append(emit("judge_mean", judge=j, value=v, n=n_j))
    if len(means) == len(JUDGES):
        lines.append(emit("judge_mean_overall", value=overall, n=len(JUDGES)))
    for i in range(len(JUDGES)):
        for k in range(i + 1, len(JUDGES)):
            lines.append(emit("judge_spearman", pair=f"{JUDGES[i]}~{JUDGES[k]}",
                              value=spearman(mat[i], mat[k]), n=len(full)))
    lines.append(emit("judge_kendall_w", value=kendall, n=len(full)))
    if stability is not None:
        lines.append(emit("judge_stability_detail",
                          value=stability, n=len(variances), note="Mean variance of repeated scores"))
    lines += episode_lines(run_dir, scores, pool)
    return lines


def episode_lines(run_dir: str | Path, scores: dict, pool: list[dict]) -> list[str]:
    from .error_evolution import _boot_slope_ci, _ols_slope
    by_w: dict[int, list[float]] = {}
    for f in pool:
        w = f.get("episode")
        key = _entry_score_key(f, scores)
        if w is None or not key:
            continue
        vals = [scores[key][j][0] for j in JUDGES if scores[key].get(j)]
        if vals:
            by_w.setdefault(int(w), []).append(sum(vals) / len(vals))
    if not by_w:
        return []
    out: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    cum: list[float] = []
    for w in sorted(by_w):
        vals = by_w[w]
        cum.extend(vals)
        inc = sum(vals) / len(vals)
        out.append(emit("judge_episode", episode=w, span="inc",
                        value=round(inc, 4), n=len(vals)))
        out.append(emit("judge_episode", episode=w, span="cum",
                        value=round(sum(cum) / len(cum), 4), n=len(cum)))
        xs.append(float(w))
        ys.append(inc)
    slope = _ols_slope(xs, ys)
    lo, hi = _boot_slope_ci(xs, ys)
    out.append(emit("judge_episode_trend",
                    value=(round(slope, 5) if slope is not None else None),
                    ci95_lo=lo, ci95_hi=hi, n_episodes=len(xs)))
    return out


def write_feature_scores(run_dir: str | Path) -> Path | None:
    run_dir = Path(run_dir)
    scores = load_scores(run_dir)
    if not scores:
        return None
    by_key = {}
    for f in read_pool(run_dir):
        if str(f.get("feature_name") or ""):
            by_key[_score_key(f)] = f
    lines = ["# Per-feature scores (metric 3: interpretability | assessed content=feature name+logic+explanation; "
             f"rubric_sha1={RUBRIC_SHA1})",
             "# metric=judge_feature feature=<name> pool_index=<pool_index> "
             "score=<three_judge_mean> njudges=<k> <judge>=<score> ... "
             "true_tier=<tier> true_iv=<value>"]
    for key in sorted(scores):
        per = {j: (sum(v) / len(v)) for j, v in scores[key].items() if v}
        if not per:
            continue
        mean = round(sum(per.values()) / len(per), 4)
        f = by_key.get(key) or {}
        kv = " ".join(f"{j.replace('-', '_')}={round(v, 4)}"
                      for j, v in sorted(per.items()))
        name = str(f.get("feature_name") or key).split("#")[0].replace(" ", "_")
        pi = f.get("pool_index")
        lines.append(f"metric=judge_feature feature={name}"
                     + (f" pool_index={pi}" if isinstance(pi, int) else "")
                     + f" score={mean} njudges={len(per)} {kv}"
                     + (f" true_tier={f.get('true_tier')}"
                        if f.get("true_tier") else "")
                     + (f" true_iv={round(float(f.get('true_iv')), 4)}"
                        if isinstance(f.get("true_iv"), (int, float)) else ""))
    path = run_dir / "judge_scores.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def judge_report(run_dir: str | Path, stability_sample: int = 20,
                 seed: int = 20260817) -> list[str]:
    return run_report(run_dir, stability_sample=stability_sample, seed=seed)


def aggregate_lines(run_dirs: list[Path], pool_limit: int = 0) -> list[str]:
    scope = (" | first %d entries" % pool_limit) if pool_limit else ""
    lines = ["# Judge aggregate (metric 3: interpretability | analysis set × three judges" + scope
             + " | rubric_sha1=" + RUBRIC_SHA1 + ")"]
    per_method: dict[str, list[tuple[str, float]]] = {}
    per_method_topk: dict[str, dict[int, list[float]]] = {}
    missing = []
    for d in sorted(run_dirs):
        scores = load_scores(d)
        if not scores:
            missing.append(d.name)
            continue
        feats = sorted(scores)
        means = {}
        for j in JUDGES:
            vals = [scores[f][j][0] for f in feats if scores[f].get(j)]
            if vals:
                means[j] = round(sum(vals) / len(vals), 4)
        mean_all = (round(sum(means.values()) / len(means), 4) if means else None)
        m = method_of(d.name)
        lines.append(emit("judge_run", run=d.name, method=m,
                          mean=mean_all, n=len(feats),
                          **{j.replace("-", "_"): v for j, v in means.items()}))
        if mean_all is not None:
            per_method.setdefault(m, []).append((d.name, mean_all))
        for k, info in sorted(topk_scores(d, pool_limit).items()):
            lines.append(emit("judge_run_topk", run=d.name, method=m, k=k,
                              value=info["value"]))
            per_method_topk.setdefault(m, {}).setdefault(k, []).append(info["value"])
    for m, rows in sorted(per_method.items()):
        vals = [v for _, v in rows]
        mean = sum(vals) / len(vals)
        std = ((sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
               if len(vals) > 1 else 0.0)
        lines.append(emit("judge_method", method=m, mean=round(mean, 4),
                          std=round(std, 4), seeds=len(vals)))
    for m, by_k in sorted(per_method_topk.items()):
        for k, vals in sorted(by_k.items()):
            mean = sum(vals) / len(vals)
            std = ((sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
                   if len(vals) > 1 else 0.0)
            lines.append(emit("judge_method_topk", method=m, k=k,
                              mean=round(mean, 4), std=round(std, 4),
                              seeds=len(vals)))
    if missing:
        lines.append(emit("judge_missing", runs=len(missing),
                          list=",".join(missing)))
    return lines


def discover_runs(runs_root: Path, methods: list[str] | None = None,
                  only: list[str] | None = None) -> tuple[list[Path], list[str]]:
    ok, empty = [], []
    for d in sorted(p for p in runs_root.iterdir() if p.is_dir()
                    and not p.name.startswith("_")):
        if methods and method_of(d.name) not in methods:
            continue
        if only and d.name not in only:
            continue
        (ok if (d / "features.json").is_file() else empty).append(d)
    return ok, [d.name for d in empty]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m experiments.judge",
        description="Offline evaluation of metric 3: interpretability (decoupled from experiments; requests → external scoring → ingestion → report)")
    ap.add_argument("run_dirs", nargs="*", help="Optional explicit run directories (default: scan --runs)")
    ap.add_argument("--runs", default="runs", help="Runs root directory (default: runs)")
    ap.add_argument("--methods", default="",
                    help="Process only these comma-separated methods (e.g. ours,static_agent,manual); empty=all")
    ap.add_argument("--only", default="", help="Process only these comma-separated run_ids")
    ap.add_argument("--stage", default="all", choices=["all", "requests", "score", "report"],
                    help="all=requests+ingestion (with --ingest)+report / requests=requests only / "
                         "score=call three judges locally (writes scores.jsonl; skips existing scores) / report=report only")
    ap.add_argument("--repeat", type=int, default=1, help="Scores per feature per judge (default: 1)")
    ap.add_argument("--stability-features", type=int, default=0,
                    help="Sample K additional features for repeated scoring to check stability; 0=disabled")
    ap.add_argument("--stability-repeat", type=int, default=3,
                    help="Repeat count for stability-sampled features (default: 3)")
    ap.add_argument("--ingest", default="", help="External scores (JSONL file or directory) to write back to each run")
    ap.add_argument("--force", action="store_true", default=True,
                    help="Overwrite existing scores.jsonl (default: overwrite; --no-force to retain)")
    ap.add_argument("--no-force", dest="force", action="store_false")
    ap.add_argument("--aggregate-out", default="", help="Aggregate output path (default: runs/judge_aggregate.txt)")
    ap.add_argument("--workers", type=int, default=4, help="Concurrent scoring workers (default: 4)")
    ap.add_argument("--limit", type=int, default=0, help="Maximum requests to score per run (0=all; useful for a trial)")
    ap.add_argument("--judges", default="", help="Override the three judges with comma-separated models (default: JUDGE_MODELS or three built-in models)")
    ap.add_argument("--rescore", action="store_true",
                    help="Rescore all requests during scoring, ignoring existing scores (required after rubric/prompt changes "
                         "to avoid mixing scores from different scales)")
    ap.add_argument("--probe", action="store_true", help="Probe judge connectivity and exit")
    ap.add_argument("--pool-limit", type=int, default=0,
                    help="Use only the first N features.json entries in file order (final paper setting: "
                         "100; top-100 covers the whole analysis set; 0=full pool)")
    args = ap.parse_args(argv)
    if args.judges:
        global JUDGES
        JUDGES = [m.strip() for m in args.judges.split(",") if m.strip()]
    if args.probe:
        ok = True
        for j in JUDGES:
            print(f"[judge][probe] {j} @ {JUDGE_API_URL} ...")
            ok = JudgeAdapter(j).probe() and ok
        raise SystemExit(0 if ok else 1)
    runs_root = Path(args.runs)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    only = [m.strip() for m in args.only.split(",") if m.strip()]
    if args.run_dirs:
        run_dirs = [Path(d) for d in args.run_dirs]
        empty: list[str] = []
    else:
        if not runs_root.is_dir():
            raise SystemExit(f"[judge] Runs root does not exist: {runs_root}")
        run_dirs, empty = discover_runs(runs_root, methods, only)
    print(f"[judge] Processing {len(run_dirs)} runs | rubric_sha1={RUBRIC_SHA1}"
          f" | judges {'/'.join(JUDGES)}"
          + (f" | first {args.pool_limit} analysis entries" if args.pool_limit else ""))
    if empty:
        print(f"[judge][skip] Runs without features.json ({len(empty)}): {empty[:8]}"
              + ("…" if len(empty) > 8 else ""))
    if not run_dirs:
        raise SystemExit("[judge] No runs to process (run experiments first, or check --methods/--only)")
    if args.stage in ("all", "requests"):
        for d in run_dirs:
            reqs = build_requests(
                d.name, d, repeat=args.repeat,
                stability_features=args.stability_features,
                stability_repeat=args.stability_repeat,
                pool_limit=args.pool_limit)
            (d / "judge_requests.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reqs),
                encoding="utf-8")
            srcs = {r["explanation_source"] for r in reqs}
            n_feat = len({r["feature_name"] for r in reqs})
            n_rep = max((int(r["repeat"]) for r in reqs), default=0) + 1
            print(f"[judge] {d.name}: {len(reqs)} requests"
                  f" ({n_feat} features × {len(JUDGES)} judges × 1~{n_rep} repeats, "
                  f"explanation sources {'/'.join(sorted(srcs))})")

        all_path = runs_root / "judge_requests_all.jsonl"
        total = 0
        with open(all_path, "w", encoding="utf-8") as fp:
            for p in sorted(runs_root.glob("*/judge_requests.jsonl")):
                rows_p = _read_jsonl(p)
                for r in rows_p:
                    fp.write(json.dumps(r, ensure_ascii=False) + "\n")
                total += len(rows_p)
        print(f"[judge] Combined requests → {all_path} ({total} entries from all runs with requests; "
              f"send to the external scorer)")
    if args.ingest:
        ingest_scores(args.ingest, runs_root, force=args.force)
    if args.stage == "score":
        for d in run_dirs:
            if not (d / "judge_requests.jsonl").is_file():
                reqs = build_requests(
                    d.name, d, repeat=args.repeat,
                    stability_features=args.stability_features,
                    stability_repeat=args.stability_repeat,
                    pool_limit=args.pool_limit)
                (d / "judge_requests.jsonl").write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reqs),
                    encoding="utf-8")
                print(f"[judge] {d.name}: generated {len(reqs)} missing requests")
        for d in run_dirs:
            score_run(d, judges=JUDGES, limit=args.limit, workers=args.workers,
                      force=args.rescore)
    if args.stage in ("all", "score", "report"):
        for d in run_dirs:
            lines = run_report(d, pool_limit=args.pool_limit)
            out = write_txt(d, "judge_report.txt", lines)
            has_scores = bool(load_scores(d))
            fs = write_feature_scores(d)
            print(f"[judge] {d.name}: report {out}"
                  + (f", per-feature scores {fs.name}" if fs else "")
                  + ("" if has_scores else " (no scores.jsonl yet; placeholder only)"))
        agg = Path(args.aggregate_out) if args.aggregate_out else (runs_root / "judge_aggregate.txt")
        lines = aggregate_lines(run_dirs, pool_limit=args.pool_limit)
        if empty:
            lines.append(emit("judge_no_pool", runs=len(empty), list=",".join(empty)))
        agg.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[judge] Aggregate → {agg} ({len(lines)} lines)")


if __name__ == "__main__":
    main(sys.argv[1:])
