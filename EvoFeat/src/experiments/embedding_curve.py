from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._common import emit, read_pool, write_txt
from .diversity_embedding import INPUT_FORMAT, _subset_mean_cosine, cosine_matrix, entry_key


OUT_NAME = "embedding_curve.txt"


def load_vectors(run_dir: str | Path) -> dict[str, list[float]]:
    path = Path(run_dir) / "embedding_diversity.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if data.get("input_format") != INPUT_FORMAT:
        return {}
    return {str(item["key"]): item["vector"] for item in data.get("features") or []
            if item.get("key") and item.get("vector")}


def compute_curve(run_dir: str | Path) -> list[str]:
    cached = load_vectors(run_dir)
    if not cached:
        return []
    by_episode: dict[int, list[dict]] = {}
    missing = 0
    for index, feature in enumerate(read_pool(run_dir, limit=0)):
        name = str(feature.get("feature_name") or "")
        episode = feature.get("episode")
        if not name or episode is None:
            continue
        vector = cached.get(entry_key(feature, index))
        if vector is None:
            missing += 1
            continue
        by_episode.setdefault(int(episode), []).append({"feature_name": name, "vector": vector})
    if missing:
        raise ValueError(f"Missing embeddings for {missing} admitted features")
    if not by_episode:
        return []
    episodes = sorted(by_episode)
    ordered = [item for episode in episodes for item in by_episode[episode]]
    matrix = cosine_matrix(ordered)
    lines: list[str] = []
    count = 0
    for episode in episodes:
        count += len(by_episode[episode])
        value = _subset_mean_cosine(matrix, list(range(count)))
        lines.append(emit("embed_pool_avg_cosine", episode=episode, span="cum",
                          value=value, n=count))
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m experiments.embedding_curve")
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args(argv)
    for run_dir in args.run_dirs:
        lines = compute_curve(run_dir)
        if not lines:
            raise ValueError(f"No complete embedding cache with episode metadata: {run_dir}")
        out = write_txt(run_dir, OUT_NAME, lines)
        print(f"[embedding_curve] {run_dir}: {len(lines)} rows -> {out.name}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
