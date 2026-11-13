# -*- coding: utf-8 -*-
"""Read-only in-memory indexing for released FraudFeat Parquet/CSV data.

The public schema describes node, edge, and label files. Vertex properties use
string comparison; numeric semantics apply to the edge attributes ``total_amount``,
``trade_count``, and ``top_k``. Amounts are stored in fen. Non-transaction edges
have one stored direction, while payer/payee transaction relations are independent
records with their own attributes.

This module provides entity lookup and adjacency queries only. Constrained-Gremlin
execution is implemented in ``gremlin_engine.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["GraphStore", "load_schema", "labels_from_frame"]


def load_schema(data_dir: str | Path) -> Dict[str, Any]:
    p = Path(data_dir) / "schema.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing public schema: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def labels_from_frame(df: pd.DataFrame, source: str | Path) -> Dict[str, int]:
    required = {"unique_id", "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Label file {source} is missing required columns: {missing}")
    if df["unique_id"].isna().any() or df["unique_id"].astype(str).eq("").any():
        raise ValueError(f"Label file {source} contains empty unique_id values")
    if df["unique_id"].duplicated().any():
        raise ValueError(f"Label file {source} contains duplicate unique_id values")
    values = pd.to_numeric(df["label"], errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise ValueError(f"Label file {source} must contain only binary 0/1 labels")
    return {str(uid): int(value) for uid, value in zip(df["unique_id"], values)}


class GraphStore:
    def __init__(self, data_dir: str | Path, *, load_edges: bool = True,
                 load_labels: bool = False,
                 node_entities: Optional[Iterable[str]] = None,
                 build_adjacency: bool = True):
        self.data_dir = Path(data_dir)
        self.schema = load_schema(self.data_dir)
        self.ext = "csv" if self.schema.get("format") == "csv" else "parquet"
        keep = set(node_entities) if node_entities else None
        self.nodes: Dict[str, pd.DataFrame] = {}
        self.entity_of: Dict[str, str] = {}
        for ent in self.schema["entities"]:
            if keep and ent["entity_type"] not in keep:
                continue
            path = self.data_dir / ent["file"]
            if not path.is_file():
                raise FileNotFoundError(f"Missing declared node file for {ent['entity_type']}: {path}")
            df = self._read(path)
            required = {"unique_id", "entity_type"}
            missing = sorted(required - set(df.columns))
            if missing:
                raise ValueError(f"Node file {path} is missing required columns: {missing}")
            if df["unique_id"].duplicated().any():
                raise ValueError(f"Node file {path} contains duplicate unique_id values")
            df = df.set_index("unique_id")
            self.nodes[ent["entity_type"]] = df
            for uid in df.index:
                self.entity_of[str(uid)] = ent["entity_type"]
        self.edges: pd.DataFrame = pd.DataFrame(
            columns=["edge_id", "src", "dst", "relation"])
        if load_edges:
            frames = []
            for rel in self.schema["relations"]:
                p = self.data_dir / rel["file"]
                if not p.is_file():
                    raise FileNotFoundError(f"Missing declared edge file for {rel.get('en')}: {p}")
                df = self._read(p)
                required = {"src", "dst", "relation"}
                missing = sorted(required - set(df.columns))
                if missing:
                    raise ValueError(f"Edge file {p} is missing required columns: {missing}")
                public_relation = str(rel.get("en") or "")
                if not df.empty and not df["relation"].astype(str).eq(public_relation).all():
                    raise ValueError(f"Edge file {p} contains relation values outside {public_relation}")
                if not df.empty:
                    frames.append(df)
            if frames:
                self.edges = pd.concat(frames, ignore_index=True)
                if "edge_id" not in self.edges.columns:
                    self.edges["edge_id"] = [f"{i + 1:020d}" for i in range(len(self.edges))]
                for col in ("total_amount", "trade_count", "top_k"):
                    if col in self.edges.columns:
                        self.edges[col] = pd.to_numeric(self.edges[col], errors="coerce")
        self._out: Dict[str, np.ndarray] = {}
        self._in: Dict[str, np.ndarray] = {}
        if build_adjacency and not self.edges.empty:
            self._build_adjacency()
        self.labels: Dict[str, int] = {}
        if load_labels:
            self.load_labels()
        self._edge_prop_cache: Dict[str, np.ndarray] = {}

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        if path.suffix == ".csv":
            return pd.read_csv(path, dtype=str, keep_default_na=False)
        return pd.read_parquet(path)

    def _build_adjacency(self) -> None:
        g = self.edges.reset_index(drop=True)
        self.edges = g
        src_groups = g.groupby("src", sort=False).indices
        dst_groups = g.groupby("dst", sort=False).indices
        self._out = {str(k): np.asarray(v, dtype=np.int64) for k, v in src_groups.items()}
        self._in = {str(k): np.asarray(v, dtype=np.int64) for k, v in dst_groups.items()}

    def load_labels(self) -> Dict[str, int]:
        declared = str((self.schema.get("labels") or {}).get("file") or "")
        p = self.data_dir / (declared or f"labels.{self.ext}")
        if not p.exists():
            p = self.data_dir / f"labels.{self.ext}"
        if not p.exists():
            raise FileNotFoundError(f"Missing label file: {p}")
        df = self._read(p)
        self.labels = labels_from_frame(df, p)
        unknown = [uid for uid in self.labels if uid not in self.entity_of]
        if unknown:
            raise ValueError(f"Label file {p} references {len(unknown)} unknown account IDs; "
                             f"first={unknown[0]}")
        return self.labels

    def has_vertex(self, uid: str) -> bool:
        return str(uid) in self.entity_of

    def vertex(self, uid: str) -> Optional[pd.Series]:
        ent = self.entity_of.get(str(uid))
        if ent is None:
            return None
        df = self.nodes.get(ent)
        if df is None:
            return None
        try:
            return df.loc[str(uid)]
        except KeyError:
            return None

    def node_value(self, uid: str, prop: str) -> Any:
        row = self.vertex(uid)
        if row is None:
            return None
        if prop in ("id", "unique_id"):
            return str(uid)
        if prop == "entity_type":
            value = row.get("entity_type")
            if value is not None and not pd.isna(value) and str(value):
                return str(value)
            return self.entity_of.get(str(uid))
        if prop == "label":
            return self.entity_of.get(str(uid))
        if prop in row.index:
            return row[prop]
        return None

    def iter_vertices(self) -> Iterable[Tuple[str, str]]:
        for ent, df in self.nodes.items():
            for uid in df.index:
                yield str(uid), ent

    def edge_rows(self, positions: Optional[np.ndarray] = None) -> pd.DataFrame:
        if positions is None:
            return self.edges
        if len(positions) == 0:
            return self.edges.iloc[0:0]
        return self.edges.iloc[positions]

    def incident(self, uid: str, direction: str = "both") -> np.ndarray:
        uid = str(uid)
        if direction == "out":
            return self._out.get(uid, np.empty(0, dtype=np.int64))
        if direction == "in":
            return self._in.get(uid, np.empty(0, dtype=np.int64))
        a = self._out.get(uid, np.empty(0, dtype=np.int64))
        b = self._in.get(uid, np.empty(0, dtype=np.int64))
        return np.concatenate([a, b]) if len(b) else a

    def edge_endpoint(self, positions: np.ndarray, side: str) -> np.ndarray:
        col = "src" if side in ("src", "other_src") else "dst"
        if len(positions) == 0:
            return np.empty(0, dtype=object)
        return self.edges.iloc[positions][col].to_numpy(dtype=object)

    def edge_values(self, positions: np.ndarray, prop: str) -> np.ndarray:
        if len(positions) == 0:
            return np.empty(0, dtype=object)
        df = self.edges.iloc[positions]
        if prop == "relation":
            return df["relation"].to_numpy(dtype=object)
        if prop in ("id", "edge_id"):
            return df["edge_id"].to_numpy(dtype=object)
        if prop in df.columns:
            return df[prop].to_numpy()
        return np.array([None] * len(df), dtype=object)

    def relation_meta(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for rel in self.schema["relations"]:
            public_name = str(rel.get("en") or "")
            for endpoint in rel.get("join_fields") or [{}]:
                if public_name:
                    out[public_name] = {
                        "src": endpoint.get("src"), "dst": endpoint.get("dst"),
                        "is_trade": bool(rel.get("is_trade")),
                        "cn": rel.get("cn"), "index": rel.get("index")}
        return out

    def summary(self) -> Dict[str, Any]:
        return {
            "entities": {e: int(len(df)) for e, df in self.nodes.items()},
            "nodes": int(sum(len(df) for df in self.nodes.values())),
            "edges": int(len(self.edges)),
            "relations": int(self.edges["relation"].nunique()) if not self.edges.empty else 0,
            "labels": len(self.labels),
        }
