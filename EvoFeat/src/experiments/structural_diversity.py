"""EvoFeat adapters for the shared FraudFeat structural-diversity protocol.

The benchmark package owns Gremlin/graph-pattern parsing, typed-graph construction,
GED, statistic signatures, and redundancy metrics. EvoFeat keeps only run-level
orchestration and cumulative experiment curves.
"""

from __future__ import annotations

from functools import lru_cache

from system.core.vocabulary import public_schema
from system.execution.parquet_backend import fraudfeat_module


@lru_cache(maxsize=1)
def _schema_index():
    return fraudfeat_module("diversity").SchemaIndex(public_schema())


def build_graph(feature):
    return fraudfeat_module("diversity").build_graph(feature, _schema_index())


def stat_signature(feature) -> tuple:
    return fraudfeat_module("diversity").stat_signature(feature, _schema_index())


def graph_edit_distance(a, b) -> int:
    return fraudfeat_module("diversity").graph_edit_distance(a, b)


def graph_similarity(a, b) -> float:
    return fraudfeat_module("diversity").graph_similarity(a, b)


def signature_key(feature) -> str:
    return fraudfeat_module("diversity").signature_key(feature, _schema_index())


def stat_similarity(a: tuple, b: tuple) -> float:
    return fraudfeat_module("diversity").stat_similarity(a, b)


def structure_similarity(a, b) -> float:
    return fraudfeat_module("diversity").structure_similarity(a, b, _schema_index())


def batch_metrics(features: list, workers: int | None = None,
                  curve_only: bool = False) -> dict:
    """Return shared metrics using the historical EvoFeat call signature.

    ``workers`` remains accepted for experiment-script compatibility. The shared
    protocol currently performs deterministic serial GED evaluation.
    """
    del workers
    result = fraudfeat_module("diversity").batch_metrics(features, _schema_index())
    if not curve_only:
        return result
    return {key: result.get(key) for key in (
        "n", "n_pairs", "distinct_structures", "avg_pairwise_sim", "near_dup_ratio",
    )}


__all__ = [
    "build_graph", "stat_signature", "graph_edit_distance", "graph_similarity",
    "signature_key", "stat_similarity", "structure_similarity", "batch_metrics",
]
