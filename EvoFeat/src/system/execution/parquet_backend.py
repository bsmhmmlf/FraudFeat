"""FraudFeat Parquet backend for online exploration and measured IV."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from .eg_plan import ID_LIST_LIMIT, sanitize_vid


AMOUNT_UNIT = "fen"
BACKEND_NAME = "fraudfeat-parquet"


def fraudfeat_module(name: str):
    """Import a module from the installed FraudFeat evaluation package."""
    try:
        return importlib.import_module(f"fraudfeat.{name}")
    except ModuleNotFoundError as exc:
        if exc.name != "fraudfeat":
            raise
        raise RuntimeError(
            "Install the FraudFeat evaluation package from the FraudFeat root: "
            "python -m pip install -e ./evaluation") from exc


def _field_signature(field: dict) -> tuple:
    return (str(field.get("name") or ""), str(field.get("dtype") or ""),
            json.dumps(field.get("values") or {}, sort_keys=True, ensure_ascii=False))


def _schema_signature(schema: dict) -> tuple:
    entities = {
        str(entity["entity_type"]): tuple(_field_signature(field)
                                    for field in entity.get("fields") or [])
        for entity in schema.get("entities") or []
    }
    relations = {
        str(relation["en"]): (
            bool(relation.get("is_trade")),
            tuple(sorted((str(endpoint.get("src") or ""), str(endpoint.get("dst") or ""))
                         for endpoint in relation.get("join_fields") or [])),
        )
        for relation in schema.get("relations") or []
    }
    return entities, relations


def validate_public_schema(data_schema: dict, runtime_schema: dict) -> None:
    # The released schema intentionally omits runtime-only metadata. The local
    # vocabulary supplies it, so a missing version inherits the runtime version
    # while an explicitly different version remains a hard error.
    data_version = data_schema.get("version", runtime_schema.get("version"))
    if data_version != runtime_schema.get("version"):
        raise ValueError("FraudFeat schema version does not match the runtime vocabulary")
    if _schema_signature(data_schema) != _schema_signature(runtime_schema):
        raise ValueError(
            "Released schema entities, attributes, relations, or endpoints do not match "
            "the EvoFeat runtime vocabulary")


def load_panel_records(data_dir: str | Path, panel_path: str | Path,
                       exclude: list[str] | None = None) -> list[dict]:
    """Load a fixed public panel while preserving its explicit class labels."""
    effectiveness = fraudfeat_module("effectiveness")
    ids, labels = effectiveness.load_eval_set(
        data_dir, exclude=exclude or [], eval_panel=str(panel_path))
    return [{"target_id": uid, "tag": "black" if labels[uid] else "white",
             "label": int(labels[uid])} for uid in ids]


class ParquetExecutor:
    """Executor facade consumed by the existing plan and measured-IV layers."""
    backend = BACKEND_NAME
    space = BACKEND_NAME
    g_prefix = "g_fraudfeat"

    def __init__(self, data_dir: str | Path, panel_path: str | Path | None = None):
        self.data_dir = Path(data_dir).expanduser().resolve()
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"FraudFeat data directory not found: {self.data_dir}")
        graph_store = fraudfeat_module("graph_store")
        engine = fraudfeat_module("gremlin_engine")
        from ..core.vocabulary import public_schema
        runtime_schema = public_schema()
        validate_public_schema(graph_store.load_schema(self.data_dir), runtime_schema)
        self.store = graph_store.GraphStore(self.data_dir, load_labels=False)
        self.store.schema.setdefault("version", runtime_schema.get("version"))
        self.store.schema.setdefault("format", runtime_schema.get("format"))
        self._execute = engine.execute_plan_local
        self.offline_panel = (load_panel_records(self.data_dir, panel_path)
                              if panel_path else None)
        self.panel_path = str(panel_path) if panel_path else None

    def preflight(self) -> bool:
        return self.store.schema.get("version") == 4

    def execute_plan_local(self, plan: dict, seed_vid: str):
        return self._execute(self.store, plan, sanitize_vid(seed_vid),
                             amount_unit=AMOUNT_UNIT, id_list_limit=ID_LIST_LIMIT)

    def submit(self, _script: str, *, space: str | None = None) -> Any:
        raise RuntimeError(
            "Raw Gremlin scripts are not accepted by the Parquet backend; "
            "execute the validated structured plan instead")

    def reset(self) -> None:
        return None

    def close(self) -> None:
        return None


def public_graph_store(data_dir: str | Path):
    return fraudfeat_module("graph_store").GraphStore(data_dir, load_labels=True)


def public_ids(path: str | Path) -> list[str]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = obj.get("labels") if isinstance(obj, dict) else obj
    return [str(uid) for uid in ids or []]
