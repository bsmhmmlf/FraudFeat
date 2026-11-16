from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ._common import read_pool, write_txt
from system.execution.parquet_backend import fraudfeat_module


BATCH = 32
BATCH_SLEEP = 0.3

_PROBE_TEXT = "graph risk feature embedding probe"


TOPK_KS = (10, 30, 50, 100)


def _shared():
    return fraudfeat_module("embedding_diversity")


def _model_name() -> str:
    return (os.environ.get("EMBEDDING_MODEL")
            or os.environ.get("FRAUDFEAT_EMBED_MODEL")
            or "Qwen3-Embedding-0.6B")


def _model_tag() -> str:
    return _model_name()


def _embed_url() -> str:
    return os.environ.get("EMBEDDING_API_URL", _shared().DEFAULT_BASE_URL)


def current_token() -> str:
    return os.environ.get("EMBEDDING_API_KEY", _shared().DEFAULT_API_KEY).strip()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Encode through the benchmark's versioned OpenAI-compatible implementation."""
    return _shared().embed_texts(
        texts, encoder="openai", model=_model_name(), base_url=_embed_url(),
        api_key=current_token(), batch=BATCH, sleep_s=BATCH_SLEEP)


def probe() -> bool:
    print(f"[embed][probe] {_embed_url()} | model={_model_name()}")
    try:
        vecs = embed_texts([_PROBE_TEXT])
    except RuntimeError as e:
        print(f"[embed][probe] failed: {e}")
        return False
    print(f"[embed][probe] OK: {len(vecs)} vector(s), dim {len(vecs[0])}")
    return True


INPUT_FORMAT = _shared().INPUT_FORMAT


def plan_text(f: dict) -> str:
    return _shared().plan_text(f)


def feature_text(f: dict) -> str:
    return _shared().feature_text(f)


def entry_key(f: dict, pos: int) -> str:
    return fraudfeat_module("pool").entry_key(f, pos)


def encode_features(run_dir: str | Path, pool_limit: int = 0) -> list[dict]:
    raw = [f for f in read_pool(run_dir, limit=pool_limit)
           if str(f.get("feature_name") or "")]
    pool = fraudfeat_module("pool").prepare_features(raw)
    texts = [feature_text(f) for f in pool]
    vecs = embed_texts(texts)
    return [{"key": entry_key(f, i), "feature_name": str(f["feature_name"]),
             "vector": [round(x, 6) for x in v]}
            for i, (f, v) in enumerate(zip(pool, vecs))]


def cosine_matrix(vectors: list[dict]) -> list[list[float]]:
    raw = [item.get("vector") or [] for item in vectors]
    return _shared().cosine_matrix(raw).tolist()


def pairwise_cosine(vectors: list[dict],
                    matrix: list[list[float]] | None = None) -> list[dict]:
    m = matrix if matrix is not None else cosine_matrix(vectors)
    names = [it.get("key") or it["feature_name"] for it in vectors]
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = m[i][j]
            out.append({"a": names[i], "b": names[j],
                        "cosine": round(c, 6), "distance": round(1 - c, 6)})
    return out


def project_2d(vectors: list[dict], method: str = "pca") -> dict:
    names = [it.get("key") or it["feature_name"] for it in vectors]
    if len(names) < 2:
        return {"method": method, "dim": 2,
                "points": [{"feature_name": n, "x": 0.0, "y": 0.0}
                           for n in names]}
    import numpy as np
    X = np.asarray([it.get("vector") or [] for it in vectors], dtype=float)
    if method == "tsne":
        from sklearn.manifold import TSNE
        coords = TSNE(n_components=2, random_state=20260817,
                      init="pca", perplexity=min(30.0, max(5.0, (len(names) - 1) / 3.0)),
                      ).fit_transform(X)
    else:
        Xc = X - X.mean(axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        coords = U[:, :2] * S[:2]
    return {"method": method, "dim": 2,
            "points": [{"feature_name": n,
                        "x": round(float(coords[i][0]), 6),
                        "y": round(float(coords[i][1]), 6)}
                       for i, n in enumerate(names)]}


def _subset_mean_cosine(matrix: list[list[float]], idx: list[int]) -> float | None:
    return _shared().pairwise_stats(matrix, idx)["avg_cosine"]


def _nn_distance(matrix: list[list[float]], idx: list[int]) -> float | None:
    return _shared().pairwise_stats(matrix, idx)["nn_distance"]


def embedding_metrics(run_dir: str | Path, pool_limit: int = 0,
                      matrix: list[list[float]] | None = None) -> list[str]:
    raw = [f for f in read_pool(run_dir, limit=pool_limit)
           if str(f.get("feature_name") or "")]
    if len(raw) < 2:
        return []
    if matrix is None:
        raise ValueError("embedding_metrics requires cosine_matrix (encode features first)")
    import numpy as np

    pool_module = fraudfeat_module("pool")
    pool = pool_module.prepare_features(raw)
    iv = {pool_module.entry_key(feature, index): float(feature["eval_iv"])
          for index, feature in enumerate(pool) if feature.get("eval_iv") is not None}
    metrics = _shared().embedding_metrics(pool, np.asarray(matrix, dtype=float), iv=iv)
    lines = [f"metric=embed_features value={metrics['embed_features']}",
             f"metric=embed_avg_cosine value={metrics['avg_cosine']}",
             f"metric=embed_nn_distance value={metrics['nn_distance']}"]
    for k in TOPK_KS:
        n = metrics[f"top{k}_embed_n"]
        if n < 2:
            continue
        lines.append(f"metric=topk_embed_avg_cosine k={k} "
                     f"value={metrics[f'top{k}_embed_avg_cosine']} n={n}")
        lines.append(f"metric=topk_embed_nn_distance k={k} "
                     f"value={metrics[f'top{k}_embed_nn_distance']} n={n}")
    return lines


def run(run_dir: str | Path, out_name: str = "embedding_diversity.json",
        pool_limit: int = 0, method: str = "pca", curve_only: bool = False) -> Path:
    run_dir = Path(run_dir)
    vectors = encode_features(run_dir, pool_limit=pool_limit)
    if curve_only:
        payload = {"run_id": run_dir.name, "model": _model_tag(),
                   "input_format": INPUT_FORMAT, "pool_limit": pool_limit,
                   "curve_only": True, "n_features": len(vectors), "features": vectors}
        path = run_dir / out_name
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"[embed] {run_dir.name}: encoded {len(vectors)} features for RQ2 curves")
        return path
    matrix = cosine_matrix(vectors)
    pairwise = pairwise_cosine(vectors, matrix=matrix)
    projection = project_2d(vectors, method=method)
    payload = {"run_id": run_dir.name, "model": _model_tag(),
               "input_format": INPUT_FORMAT, "pool_limit": pool_limit,
               "n_features": len(vectors), "features": vectors,
               "pairwise": pairwise, "projection": projection}
    path = run_dir / out_name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    lines = embedding_metrics(run_dir, pool_limit=pool_limit, matrix=matrix)
    if lines:
        write_txt(run_dir, "embedding_diversity.txt", lines)
    print(f"[embed] {run_dir.name}: encoded {len(vectors)} features"
          f" ({len(pairwise)} pairs) → {path.name}"
          + (", metric txt written" if lines else ""))
    return path


def recompute(run_dir: str | Path, pool_limit: int = 0,
              in_name: str = "embedding_diversity.json") -> bool:
    run_dir = Path(run_dir)
    path = run_dir / in_name
    if not path.is_file():
        print(f"[embed][skip] {run_dir.name}: no {in_name} (encode features first)")
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("input_format") != INPUT_FORMAT:
        print(f"[embed][skip] {run_dir.name}: cached input format "
              f"{data.get('input_format') or 'v1(name+logic+explanation)'} ≠ "
              f"{INPUT_FORMAT} (re-encode with --force)")
        return False
    cached = {str(it.get("key")): it for it in (data.get("features") or [])}
    pool = [(i, f) for i, f in enumerate(read_pool(run_dir, limit=pool_limit))
            if str(f.get("feature_name") or "")]
    missing = [entry_key(f, i) for i, f in pool if entry_key(f, i) not in cached]
    if missing:
        print(f"[embed][skip] {run_dir.name}: {len(missing)} features missing from the cache"
              f" (pool changed; re-encode with --force)")
        return False
    vectors = [cached[entry_key(f, i)] for i, f in pool]
    lines = embedding_metrics(run_dir, pool_limit=pool_limit,
                              matrix=cosine_matrix(vectors))
    if lines:
        write_txt(run_dir, "embedding_diversity.txt", lines)
    print(f"[embed][recompute] {run_dir.name}: {len(vectors)} features (cached vectors; no gateway calls)")
    return True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m experiments.diversity_embedding",
        description="Semantic diversity via feature embeddings (pairwise cosine / projection)")
    ap.add_argument("run_dirs", nargs="*", help="Run directories (runs root when --all is set)")
    ap.add_argument("--all", action="store_true",
                    help="Treat the first positional argument as the runs root; process every run with features.json")
    ap.add_argument("--pool-limit", type=int, default=0,
                    help="Use only the first N features.json entries (final paper setting: 100; 0=full pool)")
    ap.add_argument("--method", default="pca", choices=["pca", "tsne"],
                    help="2D projection method (default: pca, deterministic and reproducible)")
    ap.add_argument("--probe", action="store_true", help="Probe connectivity with one text only")
    ap.add_argument("--force", action="store_true",
                    help="Rerun even when embedding_diversity.json exists (skipped by default to avoid duplicate gateway calls)")
    ap.add_argument("--recompute", action="store_true",
                    help="Recompute embedding_diversity.txt from cached vectors using the current scope; no gateway calls")
    ap.add_argument("--curve-only", action="store_true",
                    help="Encode only the vectors needed for RQ2 cumulative semantic curves; skip other diversity metrics")
    args = ap.parse_args(argv)

    if args.probe:
        raise SystemExit(0 if probe() else 1)
    targets: list[Path] = []
    if args.all:
        if not args.run_dirs:
            raise SystemExit("[embed] --all requires the runs root directory")
        root = Path(args.run_dirs[0])
        targets = [d for d in sorted(root.iterdir()) if d.is_dir()
                   and (d / "features.json").is_file()]
    else:
        targets = [Path(d) for d in args.run_dirs]
    if not targets:
        raise SystemExit("[embed] No run directories specified (see --help for usage)")
    ok = 0
    if args.recompute:
        if args.curve_only:
            raise SystemExit("[embed] --curve-only does not use --recompute")
        for d in targets:
            ok += int(recompute(d, pool_limit=args.pool_limit))
        print(f"[embed] Recomputed {ok}/{len(targets)} runs")
        return
    for d in targets:
        cache = d / "embedding_diversity.json"
        if cache.is_file() and not args.force:
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
                expected = len(read_pool(d, limit=args.pool_limit))
                reusable = (int(data.get("pool_limit", -1)) == args.pool_limit
                            and int(data.get("n_features", -1)) == expected
                            and bool(data.get("curve_only", False)) == args.curve_only)
            except (OSError, ValueError, TypeError):
                reusable = False
            if reusable:
                print(f"[embed][skip] {d.name}: matching output already exists")
                continue
        try:
            run(d, pool_limit=args.pool_limit, method=args.method,
                curve_only=args.curve_only)
            ok += 1
        except RuntimeError as e:
            print(f"[embed][failed] {d.name}: {e}")
    print(f"[embed] Completed {ok}/{len(targets)} runs")


if __name__ == "__main__":
    main(sys.argv[1:])
