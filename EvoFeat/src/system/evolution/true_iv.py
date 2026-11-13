from __future__ import annotations

import hashlib

from ..core.schema import Candidate
from ..utils import get_logger
from .family import candidate_similarity


logger = get_logger("true_iv")


def _rel_signature(evidence_chain: list | None) -> tuple[str, ...]:
    sig: list[str] = []
    for s in (evidence_chain or []):
        if isinstance(s, dict):
            r = str(s.get("rel_name") or s.get("rel") or "").strip()
            if r:
                sig.append(r)
    return tuple(sig)


def _structure_similarity(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    if not a or not b:
        return 0.2
    if a == b:
        return 0.85
    sa, sb = set(a), set(b)
    jaccard = len(sa & sb) / len(sa | sb)
    return 0.5 if jaccard >= 0.5 else 0.2


def simulate_true_iv(cand: Candidate) -> float:
    h = int(hashlib.md5(cand.feature_name.encode()).hexdigest(), 16) % 1000 / 1000.0
    rels_text = " ".join(_rel_signature(getattr(cand, "evidence_chain", None))).lower()
    text = f"{rels_text} {cand.feature_name.lower()}"
    if any(k in text for k in ("cre_id", "bank_card")):
        base = 0.70
    elif "fan_out" in text:
        base = 0.35
    elif any(k in text for k in ("amount", "amt")):
        base = 0.25
    else:
        base = 0.10
    iv = base + (h - 0.5) * 0.10
    return round(max(0.0, iv), 4)


def compute_redundancy(cand: Candidate, pool: list[Candidate]) -> float:
    if not pool:
        return 0.0
    best = 0.0
    for p in pool:
        if p.true_iv is None:
            continue
        best = max(best, candidate_similarity(cand, p))
    return round(best, 3)


def emit_query(cand: Candidate) -> str:
    return cand.dsl_program if isinstance(cand.dsl_program, str) else ""


def _ground_iv(cand: Candidate, executor=None) -> float | None:
    try:
        from .real_iv import compute_real_iv, real_mode_enabled
        if real_mode_enabled():
            return compute_real_iv(cand, executor=executor)
    except Exception as e:
        logger.warning(f"[True IV] Real-mode computation failed ({type(e).__name__}: {e}); candidate frozen")
        return None
    iv = simulate_true_iv(cand)
    logger.info(f"[True IV] Plan B simulation stub IV={iv:.3f} (feature {cand.feature_name})")
    return iv


def compute_true_iv(cand: Candidate, pool: list[Candidate],
                   use_llm_iv: bool = False, executor=None) -> None:
    llm_iv = getattr(cand, "predicted_iv", None)
    if use_llm_iv and llm_iv is not None:
        true_iv = float(llm_iv)
        logger.info(f"[True IV] Forced evaluator LLM estimate IV={true_iv:.3f} (feature {cand.feature_name})")
    else:
        true_iv = _ground_iv(cand, executor=executor)
    if true_iv is None:
        logger.warning(f"[True IV] Measured value unavailable (feature {cand.feature_name}); candidate frozen this round"
                       f" (excluded from memory/pool/summarization)")
        return
    redundancy = compute_redundancy(cand, pool)
    cand.true_iv = true_iv
    cand.redundancy = redundancy
    cand.grounded = True
