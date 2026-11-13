"""Aggregate public run reports without recomputing evaluation metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


TOPK = (10, 30, 50, 100)


def _read(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_values(run_dir: Path) -> dict[str, float]:
    effectiveness = _read(run_dir / "effectiveness_summary.json", {})
    diversity = _read(run_dir / "diversity_summary.json", {})
    embedding_doc = _read(run_dir / "embedding_diversity.json", {})
    embedding = embedding_doc.get("metrics") or {}
    interpretability = _read(run_dir / "interpretability_summary.json", {})
    values: dict[str, float] = {}
    for key in ("valid", "invalid", "high_iv_count", "non_low_iv_count",
                "median_iv", "max_iv", "avg_coverage"):
        value = effectiveness.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[f"effectiveness.{key}"] = float(value)
    for k in TOPK:
        for source, key in ((effectiveness, f"top{k}_avg_iv"),
                            (embedding, f"top{k}_embed_avg_cosine"),
                            (embedding, f"top{k}_embed_nn_distance"),
                            (interpretability, f"top{k}_mean_score")):
            value = source.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[f"{key}"] = float(value)
        structural = diversity.get(f"top{k}") or {}
        for key in ("distinct_structures", "avg_pairwise_sim", "near_dup_ratio"):
            value = structural.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[f"top{k}_structural_{key}"] = float(value)
    for key in ("embed_avg_cosine", "embed_nn_distance"):
        value = embedding.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[key] = float(value)
    value = interpretability.get("overall_mean")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        values["interpretability.overall_mean"] = float(value)
    pool = diversity.get("pool") or {}
    for key in ("distinct_structures", "avg_pairwise_sim", "near_dup_ratio"):
        value = pool.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[f"pool_structural_{key}"] = float(value)
    return values


def _mean_std(values: list[float]) -> tuple[float, float | None]:
    mean = math.fsum(values) / len(values)
    if len(values) < 2:
        return mean, None
    std = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return mean, std


def aggregate(run_dirs: list[str | Path]) -> dict:
    runs = []
    for raw in run_dirs:
        run_dir = Path(raw).resolve()
        runs.append({"run_dir": str(run_dir), "metrics": _metric_values(run_dir)})
    keys = sorted({key for run in runs for key in run["metrics"]})
    metrics = {}
    for key in keys:
        values = [run["metrics"][key] for run in runs if key in run["metrics"]]
        mean, std = _mean_std(values)
        metrics[key] = {"n": len(values), "mean": mean, "std": std,
                        "values": values}
    return {"runs": runs, "metrics": metrics}


def render(document: dict) -> str:
    lines = ["FraudFeat Public Reproduction Aggregate",
             "=" * 72,
             f"Runs: {len(document['runs'])}", "",
             "metric | n | mean | sample_std"]
    for key, row in document["metrics"].items():
        lines.append(f"{key} | {row['n']} | {row['mean']:.6f} | "
                     f"{row['std']:.6f}" if row["std"] is not None else
                     f"{key} | {row['n']} | {row['mean']:.6f} | -")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--out", default="public_aggregate")
    args = parser.parse_args()
    document = aggregate(args.run_dirs)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".txt").write_text(render(document), encoding="utf-8")
    print(render(document), end="")


if __name__ == "__main__":
    main()
