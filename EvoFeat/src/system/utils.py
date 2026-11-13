from __future__ import annotations

import contextvars
import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from system.paths import CONFIG_DIR


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(name: str) -> dict:
    return load_yaml(CONFIG_DIR / name)


_log_file_handler: logging.FileHandler | None = None
_task_log_buffer: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "task_log_buffer", default=None)


class _TaskLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        buf = _task_log_buffer.get()
        if buf is None:
            return True
        buf.append(record)
        return False


_task_log_filter = _TaskLogFilter()


def begin_log_capture(buffer: list):
    return _task_log_buffer.set(buffer)


def end_log_capture(token) -> None:
    _task_log_buffer.reset(token)


def flush_log_records(records: list) -> None:
    for record in records:
        logging.getLogger(record.name).handle(record)


def install_file_logger(log_path: str | Path) -> logging.FileHandler:
    global _log_file_handler
    if _log_file_handler is not None:
        return _log_file_handler
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _log_file_handler = logging.FileHandler(p, encoding="utf-8")
    _log_file_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
                           "%Y-%m-%d %H:%M:%S")
    )
    _log_file_handler.setLevel(logging.INFO)
    for existing in logging.Logger.manager.loggerDict.values():
        if isinstance(existing, logging.Logger) and existing is not logging.getLogger():
            if _log_file_handler not in existing.handlers:
                existing.addHandler(_log_file_handler)
    return _log_file_handler


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"system.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
                               "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    if _log_file_handler is not None and _log_file_handler not in logger.handlers:
        logger.addHandler(_log_file_handler)
    if _task_log_filter not in logger.filters:
        logger.addFilter(_task_log_filter)
    return logger


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str | None) -> Any | None:
    if not text:
        return None

    m = _CODE_FENCE_RE.search(text)
    candidate = m.group(1) if m else text
    try:
        return json.loads(candidate)
    except Exception:
        pass
    m = _BRACE_RE.search(candidate)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def str_to_float(s: Any, default: float = 0.0) -> float:
    try:
        return float(s)
    except Exception:
        return default


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
