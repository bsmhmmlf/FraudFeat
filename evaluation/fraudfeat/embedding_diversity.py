# -*- coding: utf-8 -*-
"""Semantic diversity from feature-name, logic, query, and explanation embeddings.

Supported encoders are an OpenAI-compatible ``/embeddings`` endpoint, a local
``sentence-transformers`` model, and a deterministic character-ngram hash encoder.
The hash encoder is only an offline pipeline check, not a semantic model.

The analysis pool is selected from the input prefix before any IV ranking. Cache
entries use ``feature_name#pool_index`` so duplicate names cannot overwrite each
other. Remote responses are restored by their explicit ``index`` fields, vectors
are rounded to six decimals before cosine computation, and cache metadata must
match the current pool, encoder, model, and input format.

Examples::

    python -m fraudfeat.embedding_diversity --features features.jsonl \\
        --encoder openai --model text-embedding-3-small --out-dir results
    python -m fraudfeat.embedding_diversity --features features.jsonl \\
        --encoder hash --out-dir results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .pool import (
    analysis_pool, entry_key, load_features, load_iv_jsonl, prepare_features,
    ranked_indices,
)


__all__ = ["embed_texts", "feature_text", "plan_text", "cosine_matrix", "pairwise_stats",
           "project_2d", "embedding_metrics", "main"]
logger = logging.getLogger("fraudfeat.embedding_diversity")
DEFAULT_BASE_URL = os.environ.get("FRAUDFEAT_EMBED_BASE_URL",
                                  os.environ.get("FRAUDFEAT_JUDGE_BASE_URL",
                                                 "https://api.openai.com/v1"))
DEFAULT_MODEL = os.environ.get("FRAUDFEAT_EMBED_MODEL", "text-embedding-3-small")
DEFAULT_API_KEY = os.environ.get("FRAUDFEAT_EMBED_API_KEY",
                                 os.environ.get("FRAUDFEAT_JUDGE_API_KEY", ""))
_PROBE_TEXT = "Number of other accounts sharing a phone number with the target account"
_TOPK_KS = (10, 30, 50, 100)
INPUT_FORMAT = "name+logic+query+explanation/v3"


def _post_json(url: str, payload: Dict[str, Any], api_key: str,
               timeout: float) -> Tuple[Optional[Dict[str, Any]], str]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")), ""
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", "replace")
        except Exception:                               # noqa: BLE001
            body = ""
        return None, f"HTTP {e.code}: {body}"
    except Exception as e:                              # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _embed_openai(texts: Sequence[str], *, base_url: str, model: str, api_key: str,
                  batch: int = 32, sleep_s: float = 0.3, timeout: float = 120.0,
                  retries: int = 2) -> List[List[float]]:
    url = base_url.rstrip("/")
    url = url if url.endswith("/embeddings") else url + "/embeddings"
    out: List[List[float]] = []
    for i in range(0, len(texts), max(1, batch)):
        chunk = list(texts[i:i + max(1, batch)])
        last = ""
        for attempt in range(1 + retries):
            res, err = _post_json(url, {"model": model, "input": chunk},
                                  api_key, timeout)
            if res is not None:
                try:
                    items = list(res["data"])
                    items.sort(key=lambda d: int(d["index"]))
                    if [int(d["index"]) for d in items] != list(range(len(chunk))):
                        raise ValueError("response indices do not cover the input batch")
                    vecs = [list(map(float, d["embedding"])) for d in items]
                    if any(not v for v in vecs):
                        raise ValueError("response contains an empty embedding")
                except Exception as e:                  # noqa: BLE001
                    last = f"Response parsing failed: {e}"
                    vecs = []
                if len(vecs) == len(chunk):
                    out.extend(vecs)
                    break
                last = last or f"Unexpected response size {len(vecs)}/{len(chunk)}"
            else:
                last = err
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"Embeddings request failed ({url} model={model}): {last}")
        if sleep_s > 0 and i + len(chunk) < len(texts):
            time.sleep(sleep_s)
    return out


def _embed_sentence_transformers(texts: Sequence[str], model: str,
                                 batch: int = 32) -> List[List[float]]:
    try:
        from sentence_transformers import SentenceTransformer   # type: ignore
    except ImportError as e:                            # pragma: no cover
        raise RuntimeError(
            "sentence-transformers is not installed (pip install sentence-transformers); "
            "alternatively use --encoder openai or hash") from e
    st = SentenceTransformer(model)
    arr = st.encode(list(texts), batch_size=max(1, batch), show_progress_bar=False,
                    normalize_embeddings=False)
    return [list(map(float, row)) for row in np.asarray(arr)]


def _embed_hash(texts: Sequence[str], dim: int = 256) -> List[List[float]]:
    out: List[List[float]] = []
    for t in texts:
        vec = np.zeros(dim, dtype="float64")
        s = str(t)
        for i in range(max(1, len(s) - 2)):
            gram = s[i:i + 3]
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16)
            vec[h % dim] += 1.0
        n = float(np.linalg.norm(vec)) or 1.0
        out.append(list(vec / n))
    return out


def embed_texts(texts: Sequence[str], *, encoder: str = "openai",
                model: Optional[str] = None, base_url: str = DEFAULT_BASE_URL,
                api_key: str = DEFAULT_API_KEY, batch: int = 32,
                sleep_s: float = 0.3, timeout: float = 120.0) -> List[List[float]]:
    if not texts:
        return []
    enc = str(encoder or "openai").lower()
    if enc in ("openai", "http", "remote"):
        return _embed_openai(texts, base_url=base_url, model=model or DEFAULT_MODEL,
                             api_key=api_key, batch=batch, sleep_s=sleep_s,
                             timeout=timeout)
    if enc in ("sentence-transformers", "sentence_transformers", "st", "local"):
        return _embed_sentence_transformers(texts, model or "all-MiniLM-L6-v2", batch)
    if enc == "hash":
        return _embed_hash(texts)
    raise RuntimeError(f"Unknown encoder {encoder}; supported: openai, sentence-transformers, hash")


def plan_text(f: Dict[str, Any]) -> str:
    """Render the structured query deterministically as steps plus final JSON."""
    plan: Any = f.get("gremlin_plan") or f.get("plan") or {}
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except json.JSONDecodeError:
            return plan.strip()
    if not isinstance(plan, dict) or not plan.get("steps"):
        q = f.get("sql") or f.get("query") or f.get("dsl_program") or ""
        return q if isinstance(q, str) else json.dumps(q, ensure_ascii=False, sort_keys=True)
    lines = []
    for s in plan.get("steps") or []:
        if not isinstance(s, dict):
            continue
        lines.append(f"{s.get('id', '')}[{s.get('output', '')}]: "
                     f"{str(s.get('gremlin') or '').strip()}")
    if plan.get("final") is not None:
        lines.append("final: " + json.dumps(plan["final"], ensure_ascii=False, sort_keys=True))
    return "\n".join(lines)


def feature_text(f: Dict[str, Any]) -> str:
    """Build the versioned Name/Logic/Query/Explanation embedding input."""
    parts = [("Name", str(f.get("feature_name") or "")),
             ("Logic", str(f.get("logic") or "")),
             ("Query", plan_text(f)),
             ("Explanation", str(f.get("business_explanation") or ""))]
    return "\n".join(f"{k}: {v}" for k, v in parts if v.strip())


def _units(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    m = np.asarray(vectors, dtype="float64")
    if m.ndim != 2 or m.size == 0:
        return m
    norms = np.linalg.norm(m, axis=1)
    norms[norms == 0] = 1e-9
    return m / norms[:, None]


def cosine_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    u = _units(vectors)
    if u.ndim != 2 or u.shape[0] == 0:
        return np.zeros((0, 0))
    m = np.clip(u @ u.T, -1.0, 1.0)
    np.fill_diagonal(m, 1.0)
    return m


def pairwise_stats(matrix: np.ndarray, idx: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    m = np.asarray(matrix)
    if m.ndim != 2 or m.shape[0] < 2:
        return {"n": int(m.shape[0]) if m.ndim == 2 else 0,
                "avg_cosine": None, "nn_distance": None}
    sel = list(idx) if idx is not None else list(range(m.shape[0]))
    sub = m[np.ix_(sel, sel)]
    n = len(sel)
    if n < 2:
        return {"n": n, "avg_cosine": None, "nn_distance": None}
    iu = np.triu_indices(n, k=1)
    avg = float(np.mean(sub[iu]))
    off = sub.copy()
    np.fill_diagonal(off, -np.inf)
    nn = float(np.mean(1.0 - off.max(axis=1)))
    return {"n": n, "avg_cosine": round(avg, 6), "nn_distance": round(nn, 6)}


def project_2d(vectors: Sequence[Sequence[float]]) -> List[List[float]]:
    m = np.asarray(vectors, dtype="float64")
    if m.ndim != 2 or m.shape[0] == 0:
        return []
    centered = m - m.mean(axis=0, keepdims=True)
    try:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return [[0.0, 0.0] for _ in range(m.shape[0])]
    k = min(2, vt.shape[0])
    coords = centered @ vt[:k].T
    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 2 - coords.shape[1]))])
    return [[round(float(x), 6), round(float(y), 6)] for x, y in coords]


def _load_features(path: str | Path) -> List[Dict[str, Any]]:
    return load_features(path)


def _load_iv(path: Optional[str]) -> Dict[str, float]:
    return load_iv_jsonl(path)


def embedding_metrics(features: List[Dict[str, Any]], matrix: np.ndarray,
                      iv: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Compute whole-pool metrics and stable IV-ranked top-k subsets."""
    features = prepare_features(features)
    n = len(features)
    if np.asarray(matrix).shape != (n, n):
        raise ValueError(f"Cosine matrix shape does not match pool size {n}")
    order = ranked_indices(features, iv)
    all_stats = pairwise_stats(matrix)
    out: Dict[str, Any] = {"embed_features": n, **all_stats}
    for k in _TOPK_KS:
        idx = order[:k]
        st = pairwise_stats(matrix, idx) if len(idx) >= 2 else {"n": len(idx),
                                                                "avg_cosine": None,
                                                                "nn_distance": None}
        out[f"top{k}_embed_avg_cosine"] = st["avg_cosine"]
        out[f"top{k}_embed_nn_distance"] = st["nn_distance"]
        out[f"top{k}_embed_n"] = len(idx)
    return out


def render_report(metrics: Dict[str, Any], *, features_path: str, encoder: str,
                  model: str, dim: int) -> str:
    lines = [
        "FraudFeat Semantic Diversity Report (Feature-Text Embeddings)",
        "=" * 60,
        f"Features file       : {features_path}",
        f"Encoder             : {encoder}" + (f" (model={model})" if model else ""),
        f"Vector dimension    : {dim}",
        f"Pool feature count  : {metrics['embed_features']}",
        "",
        "Whole pool:",
        f"  Mean pairwise cosine similarity : {metrics['avg_cosine']} (higher means more homogeneous)",
        f"  Mean nearest-neighbor distance  : {metrics['nn_distance']} (higher means more diverse)",
        "",
        "Top-k (IV descending; input order when IV is unavailable):",
    ]
    for k in _TOPK_KS:
        lines.append(f"  top{k:<4}: mean_cosine={metrics.get(f'top{k}_embed_avg_cosine')} "
                     f"nn_distance={metrics.get(f'top{k}_embed_nn_distance')}")
    return "\n".join(lines) + "\n"


def _metric_lines(metrics: Dict[str, Any]) -> List[str]:
    lines = [f"metric=embed_features value={metrics['embed_features']}",
             f"metric=embed_avg_cosine value={metrics['avg_cosine']}",
             f"metric=embed_nn_distance value={metrics['nn_distance']}"]
    for k in _TOPK_KS:
        n = metrics.get(f"top{k}_embed_n")
        avg_cosine = metrics.get(f"top{k}_embed_avg_cosine")
        nn_distance = metrics.get(f"top{k}_embed_nn_distance")
        if avg_cosine is not None:
            lines.append(f"metric=topk_embed_avg_cosine k={k} "
                         f"value={avg_cosine} n={n}")
        if nn_distance is not None:
            lines.append(f"metric=topk_embed_nn_distance k={k} "
                         f"value={nn_distance} n={n}")
    return lines


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m fraudfeat.embedding_diversity",
        description="FraudFeat semantic diversity with pluggable feature-text encoders")
    ap.add_argument("--features", help="Feature JSON or JSONL file")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--encoder", default="openai",
                    choices=["openai", "sentence-transformers", "hash"])
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Encoder model name")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL")
    ap.add_argument("--api-key", default=DEFAULT_API_KEY)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--sleep", type=float, default=0.3, help="Delay between batches in seconds")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--pool-limit", type=int, default=100,
                    help="Take the first N entries as the analysis pool; 0 means all")
    ap.add_argument("--iv-jsonl", default=None,
                    help="effectiveness.jsonl used only for within-pool top-k IV ranking")
    ap.add_argument("--no-vectors", action="store_true", help="Omit raw vectors from the output")
    ap.add_argument("--force", action="store_true", help="Recompute existing output artifacts")
    ap.add_argument("--probe", action="store_true", help="Only test encoder connectivity")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    if args.probe:
        try:
            vecs = embed_texts([_PROBE_TEXT], encoder=args.encoder, model=args.model,
                               base_url=args.base_url, api_key=args.api_key,
                               timeout=args.timeout)
        except RuntimeError as e:
            logger.error("[probe] Failed: %s", e)
            return 1
        logger.info("[probe] OK: %d text(s), dimension %d, first 3 dimensions %s",
                    len(vecs), len(vecs[0]), [round(x, 4) for x in vecs[0][:3]])
        return 0
    if not args.features:
        logger.error("--features is required unless --probe is used")
        return 2
    all_features = _load_features(args.features)
    features = analysis_pool(all_features, args.pool_limit)
    if not features:
        logger.error("No evaluable features found in %s", args.features)
        return 2
    iv = _load_iv(args.iv_jsonl)
    texts = [feature_text(f) for f in features]
    texts_sha1 = hashlib.sha1(json.dumps(texts, ensure_ascii=False)
                              .encode("utf-8")).hexdigest()
    iv_sha1 = hashlib.sha1(json.dumps(iv, sort_keys=True).encode("utf-8")).hexdigest()
    expected_keys = [entry_key(f, i) for i, f in enumerate(features)]
    cache_model = args.model if args.encoder != "hash" else "hash"
    out_dir = Path(args.out_dir)
    out_json = out_dir / "embedding_diversity.json"
    doc: Optional[Dict[str, Any]] = None
    if out_json.exists() and not args.force:
        candidate = json.loads(out_json.read_text(encoding="utf-8"))
        cached_features = candidate.get("features") or []
        cached_keys = [str(f.get("key") or entry_key(f, i))
                       for i, f in enumerate(cached_features)]
        compatible = (
            candidate.get("input_format") == INPUT_FORMAT
            and candidate.get("encoder") == args.encoder
            and candidate.get("model") == cache_model
            and candidate.get("pool_limit") == args.pool_limit
            and candidate.get("n_features") == len(features)
            and candidate.get("texts_sha1") == texts_sha1
            and candidate.get("iv_sha1") == iv_sha1
            and candidate.get("vectors_included") == (not args.no_vectors)
            and cached_keys == expected_keys
            and bool(candidate.get("metrics"))
        )
        if compatible:
            doc = candidate
            logger.info("Reusing compatible cache %s", out_json)
        else:
            logger.info("Cache metadata or pool differs; recomputing %s", out_json)
    if doc is None:
        try:
            vecs = embed_texts(texts, encoder=args.encoder, model=args.model,
                               base_url=args.base_url, api_key=args.api_key,
                               batch=args.batch, sleep_s=args.sleep,
                               timeout=args.timeout)
        except RuntimeError as e:
            logger.error("%s", e)
            return 1
        vecs = [[round(float(x), 6) for x in vector] for vector in vecs]
        matrix = cosine_matrix(vecs)
        metrics = embedding_metrics(features, matrix, iv)
        doc = {
            "encoder": args.encoder, "model": cache_model,
            "input_format": INPUT_FORMAT, "pool_limit": args.pool_limit,
            "n_features": len(features), "vectors_included": not args.no_vectors,
            "texts_sha1": texts_sha1, "iv_sha1": iv_sha1,
            "dim": len(vecs[0]) if vecs else 0,
            "features": [{"key": entry_key(f, i), "entry_key": entry_key(f, i),
                          "feature_name": f["feature_name"],
                          "pool_index": f.get("pool_index"),
                          "window": f.get("window")}
                         for i, f in enumerate(features)],
            "metrics": metrics,
            "projection": project_2d(vecs),
            "pairwise": np.round(matrix, 6).tolist() if len(features) <= 300 else None,
        }
        if not args.no_vectors:
            doc["vectors"] = [{"key": entry_key(f, i),
                               "feature_name": f["feature_name"],
                               "pool_index": f.get("pool_index"), "vector": v}
                              for i, (f, v) in enumerate(zip(features, vecs))]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        (out_dir / "embedding_diversity.txt").write_text(
            "\n".join(_metric_lines(metrics)) + "\n", encoding="utf-8")
    metrics = doc.get("metrics") or {}
    dim = int(doc.get("dim") or 0)
    report = render_report(metrics, features_path=str(args.features or ""),
                           encoder=args.encoder,
                           model=args.model if args.encoder != "hash" else "",
                           dim=dim)
    (out_dir / "embedding_diversity_report.txt").write_text(report, encoding="utf-8")
    if not args.quiet:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
