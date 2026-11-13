from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.schema import Candidate, IVTier, MemoryExample
from ..paths import MEMORY_DIR, PLAYBOOK_FILES
from ..utils import ensure_dir, get_logger


logger = get_logger("evaluator_eval")
DEFAULT_EVAL_STORE = MEMORY_DIR / "eval_memory.json"
STATIC_AGENT_SAMPLE_SIZE = 60
_STATIC_METHODS = frozenset({"manual", "stat", "comm"})
_STATIC_BOOKS = PLAYBOOK_FILES


def validate_static_assets(directory: str | Path) -> int:
    directory = Path(directory)
    path = directory / "eval_memory.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"[Static-Agent] Cannot load frozen evaluator memory: {path}: {exc}") from exc
    examples = data.get("examples") if isinstance(data, dict) else None
    if (not isinstance(examples, list) or len(examples) != STATIC_AGENT_SAMPLE_SIZE
            or data.get("count") != STATIC_AGENT_SAMPLE_SIZE):
        actual = len(examples) if isinstance(examples, list) else 0
        raise SystemExit(f"[Static-Agent] Paper protocol requires exactly "
                         f"{STATIC_AGENT_SAMPLE_SIZE} measured baseline examples, found {actual} "
                         f"in {path}. Rebuild from real Manual/Stat/Comm baseline runs; "
                         "do not duplicate or fabricate examples")
    if (not isinstance(data.get("sample_seed"), int)
            or set(data.get("source_methods") or ()) != _STATIC_METHODS):
        raise SystemExit("[Static-Agent] Frozen memory lacks the reproducible sample seed "
                         "or all three prompt-baseline source methods")
    names = set()
    for index, item in enumerate(examples, 1):
        if not isinstance(item, dict):
            raise SystemExit(f"[Static-Agent] Invalid frozen memory entry {index}")
        name = str(item.get("doc_name") or "").strip()
        method = str(item.get("source_method") or "").strip()
        source = str(item.get("source_run") or "").strip()
        try:
            iv = float(item.get("true_iv"))
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"[Static-Agent] Entry {index} has no measured IV") from exc
        if (not name or name in names or not item.get("feature_desc")
                or not item.get("logic") or method not in _STATIC_METHODS
                or not (source == method or source.startswith(method + "__"))
                or not math.isfinite(iv) or iv < 0):
            raise SystemExit(f"[Static-Agent] Entry {index} lacks a unique feature, "
                             "valid measured IV, or prompt-baseline provenance")
        names.add(name)
    for book in _STATIC_BOOKS:
        if not (directory / book).is_file() or not (directory / book).stat().st_size:
            raise SystemExit(f"[Static-Agent] Missing frozen playbook: {directory / book}")
    return len(examples)


class EvaluatorMemory:
    def __init__(self, llm: Any | None = None, max_size: int | None = None,
                 store_path: str | Path | None = None,
                 persist: bool = True, reset: bool = False,
                 keep_doc_names: set | None = None, read_only: bool = False):
        self.memory: list[MemoryExample] = []
        self.max_size = max_size
        self.store_path = Path(store_path or DEFAULT_EVAL_STORE)
        self.persist = persist
        self._last_written: list[dict] = []
        self.read_only = bool(read_only)
        if self.persist or self.read_only:
            self.load()
        if reset:
            self.clear(keep_doc_names)

    def load(self) -> int:
        if not self.store_path.exists():
            logger.info(f"Evaluator memory: no history file ({self.store_path}); starting from seeds")
            return 0
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            items = (data if isinstance(data, list)
                     else (data.get("examples") or data.get("memory")
                           or data.get("items") or []))
            self.memory = [MemoryExample.from_dict(d) for d in items]
            updated = data.get("updated_at", "?") if isinstance(data, dict) else "(plain list)"
            logger.info(f"Evaluator memory: loaded {len(self.memory)} entries from {self.store_path}"
                        f" (updated_at={updated})")
        except Exception as e:
            logger.warning(f"Evaluator memory load failed ({e}); continuing with an empty store: {self.store_path}")
            self.memory = []
        return len(self.memory)

    def save(self) -> None:
        if not self.persist:
            return
        ensure_dir(self.store_path.parent)
        payload = {"updated_at": datetime.now().isoformat(timespec="seconds"),
                   "count": len(self.memory),
                   "examples": [m.to_dict() for m in self.memory]}
        self.store_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Evaluator memory: saved {len(self.memory)} entries -> {self.store_path}")

    def clear(self, keep_doc_names: set | None = None) -> None:
        if not self.persist:
            logger.info("Evaluator memory: persist=False (frozen load); reset ignored, files retained")
            return
        keep_doc_names = set(keep_doc_names or ())
        kept = ([m for m in self.memory if m.doc_name in keep_doc_names]
                if keep_doc_names else [])
        n = len(self.memory)
        self.memory = kept
        self._last_written = []
        if self.store_path.exists():
            if kept:
                self.save()
                logger.info(f"Evaluator memory: --reset retained {len(kept)} cold-start entries"
                            f" (previously {n}); organic entries cleared -> {self.store_path}")
            else:
                try:
                    self.store_path.unlink()
                except OSError as e:
                    logger.warning(f"Evaluator memory file deletion failed: {e}")
                logger.info(f"Evaluator memory: cleared (previously {n} in-memory entries; deleted {self.store_path})")
        else:
            logger.info(f"Evaluator memory: no history file; cleared {n} in-memory entries (retained {len(kept)})")

    def bootstrap(self, seeds: list[tuple]) -> None:
        for item in seeds:
            desc, iv = item[0], float(item[1])
            if len(item) >= 3 and item[2]:
                name = str(item[2]).strip()
            else:
                name = str(desc).split("|")[0].strip()
            self.memory.append(MemoryExample(desc, iv, IVTier.from_value(iv),
                                             doc_name=name))
        self._dedup()

    def remove(self, doc_names: set) -> int:
        if not doc_names:
            return 0
        before = len(self.memory)
        self.memory = [m for m in self.memory if m.doc_name not in doc_names]
        n = before - len(self.memory)
        if n:
            self.save()
            logger.info(f"Evaluator memory: removed {n} entries by doc_name (cold-start artifact cleanup)")
        return n

    def absorb(self, candidates: list[Candidate]) -> int:
        added = 0
        self._last_written = []
        for c in candidates:
            if not c.grounded or c.true_iv is None:
                continue
            desc = self._format_example(c)
            chain_compact = [
                {"from": str(s.get("from") or ""), "rel": str(s.get("rel") or ""),
                 "to": str(s.get("to") or ""), "direction": str(s.get("direction") or "")}
                for s in (getattr(c, "evidence_chain", None) or [])
                if isinstance(s, dict)]
            logic = str(getattr(c, "logic", "") or "")
            explain = str(getattr(c, "business_explanation", "") or "")
            dsl = c.dsl_program if isinstance(c.dsl_program, str) \
                else json.dumps(c.dsl_program, ensure_ascii=False, default=str)
            p_tier = str(getattr(getattr(c, "predicted_tier", None), "value", "") or "")
            p_reason = str(getattr(c, "predicted_reason", "") or "")
            exist = next((m for m in self.memory
                          if (m.doc_name and m.doc_name == c.feature_name)
                          or m.feature_desc == desc), None)
            if exist:
                exist.feature_desc = desc
                exist.true_iv = c.true_iv
                exist.tier = IVTier.from_value(c.true_iv)
                exist.predicted_iv = c.predicted_iv
                exist.predicted_conf = c.predicted_conf
                exist.predicted_tier = p_tier
                exist.predicted_reason = p_reason
                exist.doc_name = exist.doc_name or c.feature_name
                exist.logic = logic
                exist.evidence_chain = chain_compact
                exist.dsl_program = dsl
                exist.explanation = explain
                self._last_written.append(
                    {"doc_name": exist.doc_name, "is_new": False, "candidate": c})
            else:
                self.memory.append(MemoryExample(desc, c.true_iv,
                                                  IVTier.from_value(c.true_iv),
                                                  doc_name=c.feature_name,
                                                  predicted_iv=c.predicted_iv,
                                                  predicted_conf=c.predicted_conf,
                                                  logic=logic,
                                                  evidence_chain=chain_compact,
                                                  dsl_program=dsl,
                                                  explanation=explain,
                                                  predicted_tier=p_tier,
                                                  predicted_reason=p_reason))
                added += 1
                self._last_written.append(
                    {"doc_name": c.feature_name, "is_new": True, "candidate": c})
        if self._last_written:
            logger.info(f"Evaluator memory: stored {len(self._last_written)} current-episode entries"
                        f" ({added} new); source material for the evaluator calibration playbook")
        self._curate()
        return added

    @staticmethod
    def _format_example(c: Candidate) -> str:
        chain = getattr(c, "evidence_chain", None) or []
        hops = len(chain)
        structure = f"hops={hops}"
        if chain:
            notes = []
            for seg in chain:
                if isinstance(seg, dict) and seg.get("note"):
                    n = str(seg["note"]).strip()
                    if n and n not in notes:
                        notes.append(n)
            if notes:
                structure += f", chain[{' -> '.join(notes)}]"
        risk = getattr(c, "fraud_pattern", "") or c.rationale or "(no risk rationale)"
        risk = risk.replace("\n", " ")
        cn = getattr(c, "feature_name_cn", "") or ""
        name_label = f"Feature: {c.feature_name}" + (f" ({cn})" if cn else "")
        lines = [name_label, f"Structure: {structure}", f"Risk rationale: {risk}"]
        logic = str(getattr(c, "logic", "") or "").replace("\n", " ").strip()
        if logic:
            lines.append(f"Logic: {logic}")
        explain = str(getattr(c, "business_explanation", "") or "").replace("\n", " ").strip()
        if explain:
            lines.append(f"Explanation: {explain}")
        return "\n".join(lines)

    def _curate(self) -> None:
        lows = [m for m in self.memory if m.tier == IVTier.LOW]
        others = [m for m in self.memory if m.tier != IVTier.LOW]
        if self.max_size is not None and len(self.memory) > self.max_size:
            keep = (lows + others)[-self.max_size:]
            self.memory = keep

    def _dedup(self) -> None:
        seen = set()
        uniq = []
        for m in self.memory:
            if m.feature_desc not in seen:
                seen.add(m.feature_desc)
                uniq.append(m)
        self.memory = uniq

    def __len__(self):
        return len(self.memory)
