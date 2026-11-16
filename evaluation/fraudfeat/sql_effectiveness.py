#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate SQL features directly against released CSV or Parquet tables.

Each feature supplies one DuckDB SELECT/WITH query through the ``sql`` field. The
query reads the registered ``evaluation_accounts``, ``nodes``, and ``edges`` views
and returns exactly two columns, ``unique_id`` and ``value``, for every evaluation
account exactly once. ``evaluation_accounts`` contains only account IDs. Labels are
never registered in DuckDB and are used only by the Python scorer after feature
values have been returned. One query therefore computes the complete panel instead
of executing once per account.

The edge view exposes raw released values: ``total_amount`` is in RMB fen. The
additional ``total_amount_yuan`` column is ``total_amount / 100.0``. AutobinSplit,
IV, coverage, validity, pool selection, and reports use the same implementation as
the Gremlin effectiveness evaluator.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import duckdb
import numpy as np
import pandas as pd

from .autobin import AutobinSplit
from .effectiveness import load_eval_set, load_targets, render_report, summarize
from .graph_store import load_schema
from .pool import load_features


__all__ = [
    "connect_release", "validate_sql", "execute_sql_query",
    "evaluate_feature_sql", "evaluate_all_sql", "main",
]

logger = logging.getLogger("fraudfeat.sql_effectiveness")
_SQL_MASK_RE = re.compile(
    r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|--[^\n]*|/\*.*?\*/", re.S)
_FORBIDDEN_KEYWORDS_RE = re.compile(
    r"\b(?:INSERT|UPDATE|DELETE|MERGE|CREATE|DROP|ALTER|ATTACH|DETACH|COPY|"
    r"INSTALL|LOAD|PRAGMA|SET|RESET|CALL|EXPORT|IMPORT|VACUUM|CHECKPOINT)\b", re.I)
_FORBIDDEN_FUNCTIONS_RE = re.compile(
    r"\b(?:read_parquet|read_csv|read_csv_auto|read_json|read_json_objects|read_text|"
    r"read_blob|glob|getenv|sqlite_scan|postgres_scan|mysql_scan|http_get|http_post)\s*\(",
    re.I)


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _scan(path: Path) -> str:
    name = _literal(path.resolve().as_posix())
    if path.suffix.lower() == ".parquet":
        return f"read_parquet({name})"
    if path.suffix.lower() == ".csv":
        return f"read_csv({name}, header=true, all_varchar=true)"
    raise ValueError(f"Unsupported table format for SQL backend: {path}")


def _create_union_view(con: duckdb.DuckDBPyConnection, name: str,
                       paths: Sequence[Path]) -> None:
    if not paths:
        raise ValueError(f"No source files declared for view {name}")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing source table for SQL backend: {path}")
    statement = " UNION ALL BY NAME ".join(
        f"SELECT * FROM {_scan(path)}" for path in paths)
    con.execute(f"CREATE VIEW {_identifier(name)} AS {statement}")


def _columns(con: duckdb.DuckDBPyConnection, view: str) -> List[str]:
    return [str(row[0]) for row in con.execute(
        f"DESCRIBE SELECT * FROM {_identifier(view)}").fetchall()]


def connect_release(data_dir: str | Path, eval_ids: Sequence[str],
                    labels: Dict[str, int], *, threads: int = 0
                    ) -> duckdb.DuckDBPyConnection:
    """Register graph tables and label-free evaluation anchors as DuckDB views."""
    del labels  # Labels are used only by the Python scorer and are never exposed to SQL.
    root = Path(data_dir)
    schema = load_schema(root)
    entity_paths = [root / str(entity.get("file") or "") for entity in schema["entities"]]
    relation_paths = [root / str(relation.get("file") or "")
                      for relation in schema["relations"]]
    con = duckdb.connect(database=":memory:")
    try:
        if threads:
            if threads < 1:
                raise ValueError("threads must be positive or zero")
            con.execute(f"SET threads={int(threads)}")
        _create_union_view(con, "nodes", entity_paths)
        _create_union_view(con, "_fraudfeat_edges_source", relation_paths)
        node_columns = set(_columns(con, "nodes"))
        if not {"unique_id", "entity_type"}.issubset(node_columns):
            raise ValueError("The nodes view must contain unique_id and entity_type")
        edge_columns = set(_columns(con, "_fraudfeat_edges_source"))
        if not {"src", "dst", "relation"}.issubset(edge_columns):
            raise ValueError("The edges view must contain src, dst, and relation")
        def edge_column(name: str, dtype: str) -> str:
            if name in edge_columns:
                return f"CAST({_identifier(name)} AS {dtype}) AS {_identifier(name)}"
            return f"CAST(NULL AS {dtype}) AS {_identifier(name)}"
        amount_yuan = ("CAST(total_amount AS DOUBLE) / 100.0 AS total_amount_yuan"
                       if "total_amount" in edge_columns else
                       "CAST(NULL AS DOUBLE) AS total_amount_yuan")
        con.execute(
            "CREATE VIEW edges AS SELECT "
            "CAST(src AS VARCHAR) AS src, CAST(dst AS VARCHAR) AS dst, "
            "CAST(relation AS VARCHAR) AS relation, "
            + ", ".join((edge_column("total_amount", "DOUBLE"),
                         edge_column("trade_count", "BIGINT"),
                         edge_column("top_k", "BIGINT")))
            + ", " + amount_yuan + " FROM _fraudfeat_edges_source")
        panel = pd.DataFrame({
            "unique_id": pd.Series(list(eval_ids), dtype="string"),
        })
        con.register("evaluation_accounts", panel)
        return con
    except Exception:
        con.close()
        raise


def validate_sql(sql: Any) -> str:
    """Accept one read-only SELECT/WITH statement over registered views."""
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("Feature field 'sql' must be a non-empty SELECT/WITH query")
    text = sql.strip()

    def masked(source: str) -> str:
        return _SQL_MASK_RE.sub(" ", source)

    cover = masked(text)
    if ";" in cover:
        if cover.count(";") != 1 or not re.search(r";\s*$", cover):
            raise ValueError("SQL must contain exactly one statement")
        text = text.rstrip()[:-1].rstrip()
        cover = masked(text)
    match = re.match(r"\s*([A-Za-z]+)", cover)
    if not match or match.group(1).upper() not in ("SELECT", "WITH"):
        raise ValueError("Only a single SELECT or WITH query is allowed")
    forbidden = _FORBIDDEN_KEYWORDS_RE.search(cover)
    if forbidden:
        raise ValueError(f"Forbidden SQL keyword: {forbidden.group(0).upper()}")
    function = _FORBIDDEN_FUNCTIONS_RE.search(cover)
    if function:
        raise ValueError("SQL features must query registered views, not external files or services")
    return text


def execute_sql_query(con: duckdb.DuckDBPyConnection, sql: str,
                      eval_ids: Sequence[str]) -> pd.Series:
    """Execute one panel query and enforce the unique_id/value result contract."""
    query = validate_sql(sql)
    cursor = con.execute(query)
    columns = [str(item[0]).lower() for item in cursor.description]
    if columns != ["unique_id", "value"]:
        raise ValueError(
            f"SQL result columns must be exactly [unique_id, value], got {columns}")
    records = cursor.fetchall()
    expected = set(eval_ids)
    values: Dict[str, Any] = {}
    for uid, value in records:
        if uid is None:
            raise ValueError("SQL result contains a null unique_id")
        key = str(uid)
        if key in values:
            raise ValueError(f"SQL result contains duplicate unique_id: {key}")
        if key not in expected:
            raise ValueError(f"SQL result contains an account outside the evaluation panel: {key}")
        values[key] = value
    missing = expected.difference(values)
    if missing:
        raise ValueError(
            f"SQL result must return every evaluation account exactly once; missing {len(missing)}")
    series = pd.Series([values[uid] for uid in eval_ids], index=list(eval_ids))
    series = pd.to_numeric(series, errors="raise").astype(float)
    if np.isinf(series.to_numpy()).any():
        raise ValueError("SQL result contains infinite feature values")
    return series


def _error_row(feature: Dict[str, Any], eval_ids: Sequence[str],
               error: Exception) -> Dict[str, Any]:
    row = dict(feature)
    row.update({
        "iv": None, "n_bins": None, "missing_rate": None,
        "coverage": 0.0, "n_seeds": len(eval_ids), "n_zero_hit": 0,
        "n_ok": 0, "n_error": len(eval_ids),
        "exec_ok": False, "non_empty": False,
        "non_constant": False, "valid": False,
        "errors": [f"{type(error).__name__}: {error}"],
        "method": "duckdb_sql", "execution_mode": "batch_panel",
    })
    return row


def evaluate_feature_sql(con: duckdb.DuckDBPyConnection, feature: Dict[str, Any],
                         eval_ids: Sequence[str], *, labels: Dict[str, int],
                         splitter: Optional[AutobinSplit] = None) -> Dict[str, Any]:
    """Evaluate one SQL feature and apply the shared IV and validity definitions."""
    try:
        series = execute_sql_query(con, feature.get("sql"), eval_ids)
    except Exception as exc:
        return _error_row(feature, eval_ids, exc)
    splitter = splitter or AutobinSplit(smoothing_factor=0.5)
    numeric = series.to_numpy(dtype=float)
    n_seeds = len(eval_ids)
    n_zero_hit = int(np.count_nonzero(numeric == 0.0))
    non_empty = bool(n_seeds and n_zero_hit < n_seeds)
    non_constant = bool(series.nunique(dropna=True) > 1)
    row = dict(feature)
    row.update({
        "iv": None, "n_bins": None, "missing_rate": None,
        "coverage": 1.0 if n_seeds else 0.0,
        "n_seeds": n_seeds, "n_zero_hit": n_zero_hit,
        "n_ok": n_seeds, "n_error": 0,
        "exec_ok": True, "non_empty": non_empty, "non_constant": non_constant,
        "valid": False, "errors": [], "method": "duckdb_sql",
        "execution_mode": "batch_panel",
        "value_min": (round(float(np.nanmin(numeric)), 6)
                      if np.isfinite(numeric).any() else None),
        "value_max": (round(float(np.nanmax(numeric)), 6)
                      if np.isfinite(numeric).any() else None),
    })
    if not non_constant:
        row["errors"].append("constant feature values are invalid (zero variance)")
    frame = pd.DataFrame({
        "value": series.to_numpy(),
        "label": [int(labels[uid]) for uid in eval_ids],
    })
    try:
        metrics, _bins = splitter.split_evaluation(
            frame, feature_cols=["value"], label_col="label")
    except Exception as exc:
        row["errors"] = [f"Binning failed: {type(exc).__name__}: {exc}"]
        return row
    metric = metrics.get("value")
    if metric is not None:
        iv = float(metric.iv)
        if np.isfinite(iv):
            row["iv"] = round(iv, 6)
        else:
            row["errors"].append("IV is not finite")
        row["n_bins"] = int(metric.n_bins)
        row["missing_rate"] = round(float(metric.missing_rate), 6)
    row["valid"] = bool(non_empty and non_constant and row["iv"] is not None)
    return row


def evaluate_all_sql(data_dir: str | Path, features: Sequence[Dict[str, Any]],
                     eval_ids: Sequence[str], labels: Dict[str, int], *,
                     threads: int = 0) -> List[Dict[str, Any]]:
    con = connect_release(data_dir, eval_ids, labels, threads=threads)
    try:
        splitter = AutobinSplit(smoothing_factor=0.5)
        rows = []
        for feature in features:
            row = evaluate_feature_sql(con, feature, eval_ids, labels=labels,
                                       splitter=splitter)
            rows.append(row)
            logger.info("%s | IV=%s | coverage=%.4f | valid=%s",
                        row.get("feature_name"), row.get("iv"),
                        row.get("coverage") or 0.0, row.get("valid"))
        return rows
    finally:
        con.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fraudfeat.sql_effectiveness",
        description="FraudFeat effectiveness using batch DuckDB SQL over released tables")
    parser.add_argument("--data", required=True, help="Release data directory containing schema.json")
    parser.add_argument("--features", required=True, help="JSON/JSONL features with sql fields")
    parser.add_argument("--out-dir", default="results_sql")
    parser.add_argument("--exclude-targets", default=None,
                        help="Optional JSON/JSONL discovery targets to remove")
    parser.add_argument("--eval-panel", default=None,
                        help="Fixed Parquet/CSV panel with unique_id,label; excludes --limit")
    parser.add_argument("--limit", type=int, default=0,
                        help="Maximum evaluation accounts while preserving both classes; 0 means all")
    parser.add_argument("--seed", type=int, default=42, help="Panel subsampling seed")
    parser.add_argument("--pool-limit", type=int, default=0,
                        help="Input prefix used for the analysis pool; 0 means all")
    parser.add_argument("--threads", type=int, default=0,
                        help="DuckDB worker threads; 0 uses the DuckDB default")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    features = load_features(args.features)
    if not features:
        logger.error("No features found in %s", args.features)
        return 2
    exclude = load_targets(args.exclude_targets) if args.exclude_targets else []
    try:
        eval_ids, labels = load_eval_set(
            args.data, exclude=exclude, limit=args.limit, seed=args.seed,
            eval_panel=args.eval_panel)
        rows = evaluate_all_sql(args.data, features, eval_ids, labels,
                                threads=args.threads)
    except Exception as exc:
        logger.error("SQL evaluation failed: %s", exc)
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "effectiveness.jsonl").open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(rows, pool_limit=args.pool_limit)
    summary.update({
        "backend": "duckdb_sql", "execution_mode": "batch_panel",
        "duckdb_version": duckdb.__version__, "eval_size": len(eval_ids),
    })
    (out_dir / "effectiveness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render_report(
        rows, summary, data_dir=str(args.data), features_path=str(args.features),
        eval_size=len(eval_ids), excluded=len(exclude),
        amount_unit="raw fen; edges.total_amount_yuan = total_amount / 100")
    report = report.replace("FraudFeat Effectiveness Report (IV)",
                            "FraudFeat DuckDB SQL Effectiveness Report (IV)", 1)
    (out_dir / "effectiveness_report.txt").write_text(report, encoding="utf-8")
    if not args.quiet:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
