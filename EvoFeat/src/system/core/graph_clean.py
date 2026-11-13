from __future__ import annotations

import base64
import binascii
import os
import re
from typing import Any, Dict, Set, Tuple

from ..utils import get_logger
from .vocabulary import canonical_entity, entity_definitions, public_entity_fields


logger = get_logger("core.graph_clean")
_TS_ZERO_RE = re.compile(
    r"^\s*(?:0{4}-0?1-0?1|0{4}-00-00|0000/0?1/0?1|00000000|00010101)"
    r"(?:[ T](?:00:?00(?::?00)?(?:\.0+)?))?\s*(?:Z|[+-]\d{2}:?\d{2})?\s*$"
)


def is_zero_timestamp(s: str) -> bool:
    return bool(_TS_ZERO_RE.match(s))


def is_empty_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        s = v.strip().lower()
        if not s:
            return True
        if s in {"null", "none", "nan", "unknown", "unknow", "nil",
                 "n/a", "na", "-", "--", "\\\\n", "\\n"}:
            return True
        if is_zero_timestamp(v):
            return True
    return False


def is_encrypted_field(name: str) -> bool:
    n = str(name).lower()
    return ("encrypt" in n) or ("cipher" in n) or n.endswith("_enc") or ("_enc_" in n)


def is_id_or_batch_field(name: str) -> bool:
    n = str(name).lower()
    if n.endswith("_id") or n.endswith("_ids") or n.endswith("_no") or n.endswith("_code"):
        return True
    if n.endswith("_batch") or n.endswith("_batchno") or n.endswith("_batch_no") or n.endswith("_batchid"):
        return True
    return False


def is_noise_field_name(name: str) -> bool:
    n = str(name).lower()
    if "time_stamp" in n or "timestamp" in n:
        return True
    if n in {"rowkey", "row_key"}:
        return True
    return False


_BASE64_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/=_-"
)


def is_ciphertext_value(name: str, value: Any, threshold: int = 24) -> bool:
    if not isinstance(value, str):
        return False
    s = value.strip()
    if len(s) < threshold:
        return False
    if any(ch not in _BASE64_CHARS for ch in s):
        return False
    try:
        t = s.replace("-", "+").replace("_", "/")
        pad = (-len(t)) % 4
        if pad:
            t += "=" * pad
        base64.b64decode(t, validate=False)
        return True
    except (binascii.Error, ValueError):
        return False


_MAX_VALUE_LEN = 30
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_LOG_RE = re.compile(r"^\s*\[[^\]]+\]\s*\[[^\]]+\]\s*.+")
_B64_CHARS_RE = re.compile(r"^[A-Za-z0-9+/=_-]+$")


def is_oversized_value(name: str, value: Any) -> bool:
    if not isinstance(value, str):
        return False
    s = value.strip()
    if len(s) > 128:
        return True
    if len(s) > _MAX_VALUE_LEN and (_URL_RE.search(s) or _LOG_RE.search(s)):
        return True
    if len(s) > 64 and _B64_CHARS_RE.fullmatch(s):
        return True
    return False


def load_prefix_blacklist(path: str | None) -> Set[str]:
    out: Set[str] = set()
    if not path or str(path).lower() == "none" or not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line and not line.startswith("#"):
                out.add(line)
    return out


def load_edge_field_blacklist(path: str | None) -> Tuple[Set[str], Set[str], Set[str]]:
    both: Set[str] = set()
    attr: Set[str] = set()
    edge: Set[str] = set()
    if not path or str(path).lower() == "none" or not os.path.isfile(path):
        return both, attr, edge
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if " @attr" in line:
                attr.add(line.replace("@attr", "").strip())
            elif " @edge" in line:
                edge.add(line.replace("@edge", "").strip())
            else:
                both.add(line)
    return both, attr, edge


VERTEX_META_FIELDS = {"vtype", "source_table", "attributes", "~id", "~label"}


class GraphCleaner:
    def __init__(self, *, vtype_blacklist_path: str | None = None,
                 vid_blacklist_path: str | None = None,
                 field_blacklist_path: str | None = None):
        self.vtype_bl: Set[str] = load_prefix_blacklist(vtype_blacklist_path)
        self.vid_bl: Set[str] = load_prefix_blacklist(vid_blacklist_path)
        bl_both, bl_attr, _bl_edge = load_edge_field_blacklist(field_blacklist_path)
        self.attr_bl: Set[str] = bl_both | bl_attr
        logger.info(
            f"[blacklist] Loaded {len(self.vtype_bl)} vtype prefixes / "
            f"{len(self.vid_bl)} vid prefixes / {len(self.attr_bl)} attribute fields")

    def vtype_blocked(self, vtype: str) -> bool:
        value = (vtype or "").strip()
        entity = canonical_entity(value)
        if entity not in entity_definitions():
            return True
        return any(value.startswith(prefix) or entity.startswith(prefix)
                   for prefix in self.vtype_bl)

    def vid_blocked(self, hub_vid: str) -> bool:
        v = (hub_vid or "").strip()
        return bool(v) and any(v.startswith(p) for p in self.vid_bl)

    def clean_attrs(self, entity: str, attrs: Dict[str, Any] | None) -> Dict[str, Any]:
        if not isinstance(attrs, dict) or not attrs:
            return {}
        allowed = public_entity_fields(entity)
        if not allowed:
            return {}
        out: Dict[str, Any] = {}
        for k, v in attrs.items():
            ks = str(k)
            if ks not in allowed:
                continue
            if ks in VERTEX_META_FIELDS or ks.startswith("~"):
                continue
            if ks in self.attr_bl:
                continue
            if is_encrypted_field(ks) or is_id_or_batch_field(ks) or is_noise_field_name(ks):
                continue
            if is_empty_value(v):
                continue
            if is_ciphertext_value(ks, v) or is_oversized_value(ks, v):
                continue
            out[ks] = v
        return out
