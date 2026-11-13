from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


MAX_FEATURE_HOPS = 5


class IVTier(str, Enum):
    HIGH = "high"
    MID = "mid"
    LOW = "low"

    @classmethod
    def from_value(cls, v: float) -> "IVTier":
        if v >= 0.5:
            return cls.HIGH
        if v >= 0.2:
            return cls.MID
        return cls.LOW

    def __lt__(self, other: "IVTier") -> bool:
        order = {cls.LOW: 0, cls.MID: 1, cls.HIGH: 2}
        return order[self] < order[other]


@dataclass
class TraceStep:
    action: str
    obs: str = ""
    thought: str = ""
    evidence: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"action": self.action, "obs": self.obs, "thought": self.thought}


@dataclass
class ExplorerState:
    visited: set[str] = field(default_factory=set)
    discarded: dict = field(default_factory=dict)
    probed: set[str] = field(default_factory=set)
    focus: str | None = None
    action_log: list[dict] = field(default_factory=list)
    round: int = 0

    def add_log(self, hop: int, action: str, ids: list, why: str = "") -> None:
        self.action_log.append({"hop": hop, "action": action,
                                "ids": list(ids), "why": why})

    def to_dict(self) -> dict:
        return {
            "visited": sorted(self.visited),
            "discarded": dict(self.discarded),
            "probed": sorted(self.probed),
            "focus": self.focus,
            "action_log": list(self.action_log),
            "round": self.round,
        }


FRAUD_PATTERN_FIELDS = (
    "feature_name", "feature_name_cn", "fraud_pattern", "evidence_chain",
    "path", "hops", "rationale", "expected_value_type",
)


@dataclass
class Candidate:
    feature_name: str
    dsl_program: str | dict
    value_type: str
    rationale: str = ""
    binning: str = "monotonic"
    predicted_tier: IVTier = IVTier.LOW
    predicted_conf: float = 0.0
    predicted_iv: float | None = None
    trace: list[TraceStep] = field(default_factory=list)
    episode: int | None = None

    business_explanation: str = ""

    true_iv: float | None = None
    true_miv: float | None = None
    redundancy: float | None = None
    reward: float | None = None

    grounded: bool = False
    synth_ok: bool = True
    validation_report: str = ""

    evidence: list = field(default_factory=list)
    center_vid: str = ""

    feature_name_cn: str = ""

    fraud_pattern: str = ""
    evidence_chain: list = field(default_factory=list)
    expected_value_type: str = ""
    logic: str = ""
    consistency_report: str = ""

    gremlin_plan: dict = field(default_factory=dict)
    feature_value: Any = None
    execution_trace: list = field(default_factory=list)
    exec_error: str = ""
    exec_attempts: int = 0
    consistency_detail: dict = field(default_factory=dict)

    def __hash__(self):
        return hash(self.feature_name)

    def to_dict(self) -> dict:
        return {
            "feature_name": self.feature_name,
            "feature_name_cn": self.feature_name_cn,
            "dsl_program": self.dsl_program,
            "value_type": self.value_type,
            "rationale": self.rationale,
            "binning": self.binning,
            "predicted_tier": self.predicted_tier.value,
            "predicted_conf": round(self.predicted_conf, 3),
            "episode": self.episode,
            "business_explanation": self.business_explanation,
            "predicted_iv": None if self.predicted_iv is None else round(self.predicted_iv, 4),
            "true_iv": None if self.true_iv is None else round(self.true_iv, 4),
            "true_miv": None if self.true_miv is None else round(self.true_miv, 4),
            "redundancy": None if self.redundancy is None else round(self.redundancy, 3),
            "reward": None if self.reward is None else round(self.reward, 4),
            "grounded": self.grounded,
            "synth_ok": self.synth_ok,
            "evidence": self.evidence,
            "center_vid": self.center_vid,
            "fraud_pattern": self.fraud_pattern,
            "evidence_chain": self.evidence_chain,
            "expected_value_type": self.expected_value_type,
            "logic": self.logic,
            "consistency_report": self.consistency_report,
            "gremlin_plan": self.gremlin_plan,
            "feature_value": self.feature_value,
            "execution_trace": self.execution_trace,
            "exec_error": self.exec_error,
            "exec_attempts": self.exec_attempts,
            "consistency_detail": self.consistency_detail,
        }

    @staticmethod
    def from_dict(d: dict) -> "Candidate":
        c = Candidate(
            feature_name=d["feature_name"],
            dsl_program=d["dsl_program"],
            value_type=d["value_type"],
            rationale=d.get("rationale", ""),
            binning=d.get("binning", "monotonic"),
            predicted_tier=IVTier(d.get("predicted_tier", "low")),
            predicted_conf=float(d.get("predicted_conf", 0.0)),
        )
        for k in ("true_iv", "true_miv", "redundancy", "reward", "predicted_iv"):
            if d.get(k) is not None:
                setattr(c, k, float(d[k]))
        c.episode = d.get("episode", d.get("window"))
        c.business_explanation = str(d.get("business_explanation") or "")
        c.grounded = bool(d.get("grounded", False))
        c.synth_ok = bool(d.get("synth_ok", True))
        c.evidence = d.get("evidence", []) or []
        c.center_vid = d.get("center_vid", "")
        c.feature_name_cn = d.get("feature_name_cn", "")
        c.fraud_pattern = d.get("fraud_pattern", d.get("risk_pattern", ""))
        c.evidence_chain = d.get("evidence_chain", []) or []
        c.expected_value_type = d.get("expected_value_type", "")
        c.logic = d.get("logic", "")
        c.consistency_report = d.get("consistency_report", "")
        c.gremlin_plan = d.get("gremlin_plan", {}) or {}
        c.feature_value = d.get("feature_value")
        c.execution_trace = d.get("execution_trace", []) or []
        c.exec_error = d.get("exec_error", "")
        c.exec_attempts = int(d.get("exec_attempts", 0) or 0)
        c.consistency_detail = d.get("consistency_detail", {}) or {}
        return c


@dataclass
class Skill:
    id: str
    when: str
    procedure: list[str]
    anchor: str = ""
    stage: str = "hypothesize"
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "when": self.when,
            "procedure": self.procedure, "anchor": self.anchor,
            "stage": self.stage, "stats": self.stats,
        }

    @staticmethod
    def from_dict(d: dict) -> "Skill":
        old = dict(d.get("stats") or {})
        stats = {
            "revised": int(old.get("revised", old.get("updates", 0)) or 0),
            "last_rev": int(old.get("last_rev", 0) or 0),
            "episode": int(old.get("episode", old.get("win", 0)) or 0),
        }
        return Skill(
            d["id"], d["when"], d.get("procedure", []),
            d.get("anchor", "") or d.get("yields_motif", ""),
            d.get("stage", "hypothesize"),
            stats)


@dataclass
class DslTemplate:
    id: str
    pattern: str
    slots: dict = field(default_factory=dict)
    prior: str = ""
    induced: bool = False

    def to_dict(self) -> dict:
        return {"id": self.id, "pattern": self.pattern,
                "slots": self.slots, "prior": self.prior,
                "induced": self.induced}


@dataclass
class MemoryExample:
    feature_desc: str
    true_iv: float
    tier: IVTier
    doc_name: str = ""
    predicted_iv: float | None = None
    predicted_conf: float | None = None
    logic: str = ""
    evidence_chain: list = field(default_factory=list)
    dsl_program: str = ""
    explanation: str = ""
    predicted_tier: str = ""
    predicted_reason: str = ""

    def calibration_tag(self) -> str:
        if self.predicted_iv is None:
            return ""
        gap = float(self.predicted_iv) - float(self.true_iv)
        if gap > MemoryExample.CALIBRATION_TOL:
            return "over-calibrated (overestimated)"
        if gap < -MemoryExample.CALIBRATION_TOL:
            return "under-calibrated (underestimated)"
        return "well calibrated"

    CALIBRATION_TOL = 0.05

    def to_prompt(self) -> str:
        line = (f"{self.feature_desc}\n"
                f"Measured IV={self.true_iv:.3f} (tier={self.tier.value})")
        if self.predicted_iv is not None:
            gap = float(self.predicted_iv) - float(self.true_iv)
            line += (f"\nEarlier prediction: IV={self.predicted_iv:.3f}"
                     + (f", tier={self.predicted_tier}" if self.predicted_tier else "")
                     + (f", confidence={self.predicted_conf:.2f}"
                        if self.predicted_conf is not None else ""))
            line += f"\nCalibration: {self.calibration_tag()} (prediction - measured = {gap:+.3f})"
        elif self.predicted_conf is not None:
            line += f"\nEarlier confidence={self.predicted_conf:.2f}"
        if self.predicted_reason:
            line += f"\nEarlier assessment justification: {self.predicted_reason}"
        rels = _chain_rel_short(self.evidence_chain)
        if rels:
            line += f"\nRelation sequence: {' -> '.join(rels)}"
        dsl = str(self.dsl_program or "").strip()
        if dsl:
            line += f"\nQuery: {dsl}"
        return line

    def to_embed_text(self) -> str:
        line = self.feature_desc
        rels = _chain_rel_short(self.evidence_chain)
        if rels:
            line += f"\nRelation sequence: {' -> '.join(rels)}"
        return line

    def to_dict(self) -> dict:
        return {"feature_desc": self.feature_desc,
                "true_iv": self.true_iv,
                "tier": self.tier.value,
                "doc_name": self.doc_name,
                "predicted_iv": self.predicted_iv,
                "predicted_conf": self.predicted_conf,
                "predicted_tier": self.predicted_tier,
                "predicted_reason": self.predicted_reason,
                "logic": self.logic,
                "evidence_chain": list(self.evidence_chain or []),
                "dsl_program": self.dsl_program,
                "explanation": self.explanation}

    @staticmethod
    def from_dict(d: dict) -> "MemoryExample":
        return MemoryExample(
            feature_desc=d["feature_desc"],
            true_iv=float(d["true_iv"]),
            tier=IVTier(d["tier"]),
            doc_name=d.get("doc_name", ""),
            predicted_iv=d.get("predicted_iv"),
            predicted_conf=d.get("predicted_conf"),
            logic=str(d.get("logic") or ""),
            evidence_chain=list(d.get("evidence_chain") or []),
            dsl_program=str(d.get("dsl_program") or ""),
            explanation=str(d.get("explanation") or ""),
            predicted_tier=str(d.get("predicted_tier") or ""),
            predicted_reason=str(d.get("predicted_reason") or ""),
        )


def _chain_rel_short(chain) -> list[str]:
    out: list[str] = []
    for s in (chain or []):
        if not isinstance(s, dict):
            continue
        rel = str(s.get("rel") or s.get("rel_name") or "").strip()
        if rel:
            out.append(rel)
            continue
        legacy = str(s.get("join_field") or s.get("jf") or "").strip()
        if legacy:
            left, _, right = legacy.partition("|")
            segs = [x.strip() for x in left.split("&") if x.strip()]
            short = "&".join(segs[:2])
            tail = right.strip()
            out.append(f"{short}|{tail}" if tail in ("payer", "payee") else short)
    return out
