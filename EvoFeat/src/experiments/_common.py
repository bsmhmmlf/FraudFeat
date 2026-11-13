from __future__ import annotations

import json
from pathlib import Path


def emit(name: str, **kvs) -> str:
    parts = [f"metric={name}"]
    for k, v in kvs.items():
        if v is None:
            continue
        if isinstance(v, float):
            v = round(v, 4)
        parts.append(f"{k}={v}")
    return " ".join(parts)


def parse_line(line: str) -> tuple[str, dict] | None:
    line = line.strip()
    if not line.startswith("metric="):
        return None
    parts = line.split(" ")
    name = parts[0].split("=", 1)[1]
    kvs = {}
    for p in parts[1:]:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        kvs[k] = v
    return name, _normalize_episode(kvs)


def _normalize_episode(record: dict) -> dict:
    if "episode" not in record and "window" in record:
        record["episode"] = record["window"]
    record.pop("window", None)
    if "episode_stats" not in record and "window_stats" in record:
        record["episode_stats"] = record["window_stats"]
    record.pop("window_stats", None)
    return record


def effective_iv(v, run_dir: str | Path | None) -> float | None:
    if not isinstance(v, (int, float)):
        return None
    return float(v)


def rank_by_iv(pool: list[dict], run_dir: str | Path | None = None) -> list[dict]:
    def _key(f: dict) -> float:
        v = effective_iv(f.get("eval_iv"), run_dir)
        return -v if v is not None else float("inf")

    return sorted(pool, key=_key)


def read_candidates(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / "candidates.jsonl"
    if not path.is_file():
        return []
    by_name: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = _normalize_episode(json.loads(line))
        except json.JSONDecodeError:
            continue
        if rec.get("feature_name"):
            by_name[rec["feature_name"]] = rec
    return list(by_name.values())


_OPEN_OUTCOMES = ("truncated", "queued")


def read_candidate_rows(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / "candidates.jsonl"
    if not path.is_file():
        return []
    rows: list[dict] = []
    open_idx: dict[tuple, list[int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = _normalize_episode(json.loads(line))
        except json.JSONDecodeError:
            continue
        name = rec.get("feature_name")
        if not name:
            continue
        key = (name, rec.get("episode"))
        outcome = rec.get("outcome")
        waiting = open_idx.get(key) or []
        if outcome not in _OPEN_OUTCOMES and waiting:
            rows[waiting.pop(0)] = rec
            continue
        rows.append(rec)
        if outcome in _OPEN_OUTCOMES:
            open_idx.setdefault(key, []).append(len(rows) - 1)
    return rows


def reconcile_outcomes(rows: list[dict], pool: list[dict]) -> tuple[list[dict], int]:
    out = [dict(r, outcome_raw=r.get("outcome")) for r in rows]
    by_key: dict[tuple, list[int]] = {}
    for i, r in enumerate(out):
        by_key.setdefault((r.get("feature_name"), r.get("episode")), []).append(i)
    pref = {"pooled": 0, "truncated": 1, "queued": 1, "frozen": 2, "pruned_low": 3}
    matched: set[int] = set()
    unmatched = 0
    for f in pool:
        idxs = [i for i in by_key.get((f.get("feature_name"), f.get("episode")), [])
                if i not in matched and out[i].get("outcome") in pref]
        if not idxs:
            unmatched += 1
            continue
        i = min(idxs, key=lambda j: (pref[out[j]["outcome"]], j))
        matched.add(i)
        out[i]["outcome"] = "pooled"
        for k in ("true_iv", "predicted_iv"):
            if out[i].get(k) is None and f.get(k) is not None:
                out[i][k] = f[k]
    for i, r in enumerate(out):
        if r.get("outcome") == "pooled" and i not in matched:
            if r.get("death_stage"):
                r["outcome"] = "synth_failed"
            elif r.get("stratum") and r.get("true_iv") is None:
                r["outcome"] = "frozen"
            else:
                r["outcome"] = "truncated"
    return out, unmatched


def read_candidates_reconciled(run_dir: str | Path) -> list[dict]:
    rows, _ = reconcile_outcomes(read_candidate_rows(run_dir), read_pool(run_dir))
    return rows


def analysis_cutoff(pool_all: list[dict], limit: int) -> int | None:
    if not limit or limit <= 0:
        return None
    ws = [int(f["episode"]) for f in pool_all[:limit] if f.get("episode") is not None]
    return max(ws) if ws else None


def within_cutoff(rec: dict, cutoff: int | None) -> bool:
    if cutoff is None or rec.get("episode") is None:
        return True
    return int(rec["episode"]) <= cutoff


def read_pool(run_dir: str | Path, limit: int = 0) -> list[dict]:
    path = Path(run_dir) / "features.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        feats = data.get("features") or []
    else:
        feats = data if isinstance(data, list) else []
    feats = [_normalize_episode(feature) for feature in feats]
    return feats[:limit] if limit and limit > 0 else feats


def read_history(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / "progress.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [_normalize_episode(record) for record in (data.get("history") or [])]


def read_usage(run_dir: str | Path) -> dict:
    path = Path(run_dir) / "progress.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("usage") or {}


def write_txt(run_dir: str | Path, name: str, lines: list[str]) -> Path:
    path = Path(run_dir) / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
