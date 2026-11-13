from __future__ import annotations

import json
from typing import Any

from ..llm.client import LLMClient
from ..parallel import run_parallel
from ..utils import get_logger


logger = get_logger("trajectory_digest")
DEFAULT_MAX_WORKERS = 10


_KEYS = ("structure", "signals", "nav_path", "statistics",
         "risk", "failure", "boundary", "do", "avoid")
_POS_SYSTEM = """Summarize one complete successful feature trajectory for reusable exploration knowledge. Cover exploration, hypothesis generation, synthesis, evaluation, and measured IV without dropping useful evidence.
Return detailed English JSON with keys structure, signals, nav_path, statistics, risk, failure, boundary, do, and avoid.
structure: transferable graph shape and hops; signals: observable entity, sharing, attribute, amount, and count cues; nav_path: successful action order and structural completeness; statistics: metric, filters, reduction, anchoring, normal versus fraud behavior, and threshold rationale; risk: plain business meaning; failure: potential pitfalls; boundary: applicability; do and avoid: transferable practices.
Do not include concrete node IDs, absolute amounts, or one-off instance details. Return JSON only."""

_NEG_SYSTEM = """Summarize one complete failed or low-value feature trajectory as reusable negative knowledge. Explain the failure mechanism rather than giving only a conclusion.
Return detailed English JSON with keys structure, signals, nav_path, statistics, risk, failure, boundary, do, and avoid. Cover the intended graph shape, ignored signals, incorrect navigation branch, statistical or grounding defect, remaining risk meaning, measured or validation failure reason, applicability boundary, and better practice.
Do not include concrete node IDs, absolute amounts, or one-off instance details. Return JSON only."""


def _json(v: Any, indent: int = 0) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, indent=indent, default=str)
    return str(v)


def _material_text(ep: dict) -> str:
    parts: list[str] = []
    name = ep.get("feature") or "?"
    cn = ep.get("feature_cn") or ""
    name_txt = f"{name} ({cn})" if cn else name
    head = [f"Feature: {name_txt}"]
    if ep.get("value_type"):
        head.append(f"value_type={ep.get('value_type')}")
    if ep.get("expected_value_type"):
        head.append(f"expected_value_type={ep.get('expected_value_type')}")
    if ep.get("hops"):
        head.append(f"hops={ep.get('hops')}")
    label = "positive high-value" if ep.get("positive") else "negative low-value"
    head.append(f"Label: {label}")
    if ep.get("unverified"):
        head.append("unverified")
    if ep.get("verified_pruned"):
        head.append("verified after pruning")
    if ep.get("forced"):
        head.append("forced label")
    parts.append(" | ".join(head))
    fraud_pattern = ep.get("fraud_pattern") or ep.get("risk_pattern")
    if fraud_pattern:
        parts.append(f"Fraud pattern: {fraud_pattern}")
    if ep.get("rationale"):
        parts.append(f"Risk rationale: {ep['rationale']}")
    if ep.get("logic"):
        parts.append(f"Full logic:\n{ep['logic']}")
    chain = ep.get("evidence_chain") or []
    if chain:
        rows = []
        for i, s in enumerate(chain, 1):
            if not isinstance(s, dict):
                rows.append(f"  {i}. {s}")
                continue
            rel = s.get("rel") or s.get("rel_name") or s.get("join_field") or "?"
            note = s.get("note") or ""
            rows.append(f"  {i}. {s.get('from', '?')} -[{rel}]-> {s.get('to', '?')}"
                        f" (hop={s.get('hop', i)}"
                        + (f" | {s.get('direction')}" if s.get("direction") else "")
                        + ")" + (f": {note}" if note else ""))
        parts.append("Evidence chain:\n" + "\n".join(rows))
    parts.append("Evaluation and measured value:")
    ev = []
    if ep.get("predicted_tier") or ep.get("predicted_iv") is not None:
        ev.append(f"Predicted tier={ep.get('predicted_tier')}"
                  f" iv={ep.get('predicted_iv')} conf={ep.get('predicted_conf')}")
    if ep.get("true_iv") is not None:
        ev.append(f"Measured value iv={ep.get('true_iv')}"
                  + (f" iv={ep.get('iv', ep.get('miv'))}"
                     if ep.get('iv', ep.get('miv')) is not None else "")
                  + (f" reward={ep.get('reward')}" if ep.get("reward") is not None else "")
                  + (f" redundancy={ep.get('redundancy')}"
                     if ep.get("redundancy") is not None else ""))
    if ep.get("predicted_reason"):
        ev.append(f"Evaluation basis (full text): {ep['predicted_reason']}")
    parts.append("\n".join(ev) or "  (no evaluation or measured-value record)")
    if ep.get("consistency_report") or ep.get("consistency_detail"):
        parts.append("Consistency:")
        if ep.get("consistency_report"):
            parts.append(f"  report: {ep['consistency_report']}")
        if ep.get("consistency_detail"):
            parts.append(f"  detail: {_json(ep['consistency_detail'])}")
    if ep.get("exec_error"):
        parts.append(f"Execution error: {ep['exec_error']}")
    if ep.get("feature_value") is not None and str(ep.get("feature_value")) != "None":
        parts.append(f"Execution value: {ep.get('feature_value')}")
    if ep.get("skills_used"):
        parts.append(f"Declared skills: {ep['skills_used']}")
    err_hist = ep.get("err_history") or []
    if err_hist:
        parts.append("Accumulated failure history by round (full text):")
        parts += [f"  · {x}" for x in err_hist]
    if ep.get("failed_plan"):
        parts.append(f"Last failed plan (full text):\n{_json(ep['failed_plan'], indent=1)}")
    trace = ep.get("trace") or []
    if trace:
        steps = []
        for i, t in enumerate(trace, 1):
            act = str(t.get("action", "") or "")
            obs = str(t.get("obs", "") or "")
            th = str(t.get("thought", "") or "")
            seg = f"  {i}. {act}"
            if obs:
                seg += f"\n     observation: {obs}"
            if th:
                seg += f"\n     reason: {th}"
            steps.append(seg)
        parts.append("Trajectory (action-observation-reason, full text):\n" + "\n".join(steps))
    return "\n\n".join(parts)


def digest_one(ep: dict, llm: LLMClient) -> dict | None:
    if not (ep.get("positive") or ep.get("negative")):
        return None
    system_p = _POS_SYSTEM if ep.get("positive") else _NEG_SYSTEM
    kind = "positive high-value" if ep.get("positive") else "negative low-value"
    user = (f"This is the complete, unmodified trajectory for one {kind} feature. "
            f"Read it fully and return the summary JSON required by the system instructions.\n\n{_material_text(ep)}")
    resp = llm.complete_json(system_p, user, temperature=0.3)
    if not isinstance(resp, dict) or not resp:
        return None
    return {k: str(resp.get(k, "") or "").strip() for k in _KEYS}


def digest_trajectories(episodes: list[dict], llm: LLMClient | None = None,
                        max_workers: int = DEFAULT_MAX_WORKERS) -> list[dict | None]:
    tasks = [(i, e) for i, e in enumerate(episodes)
             if e.get("positive") or e.get("negative")]
    if not tasks:
        logger.info("[Digest] No positively or negatively labeled trajectories; skipping feature-level digests (zero LLM calls)")
        return [None] * len(episodes)

    llm = llm or LLMClient()
    logger.info(f"[Digest] Full feature-trajectory digests: {len(tasks)} entries"
                f" (positive={sum(1 for _, e in tasks if e.get('positive'))}"
                f"/negative={sum(1 for _, e in tasks if e.get('negative'))}, "
                f"concurrency={max_workers})")

    def _worker(task: tuple[int, dict], _idx: int):
        i, ep = task
        try:
            return i, digest_one(ep, llm)
        except Exception as e:
            logger.warning(f"[Digest] Trajectory {i + 1} digest failed ({e}); skipping this entry")
            return i, None

    results = run_parallel(tasks, _worker, max_workers=max_workers,
                           stage="Feature trajectory digests (separate positive/negative prompts)")
    digests: list[dict | None] = [None] * len(episodes)
    ok = 0
    for res in results:
        if not res:
            continue
        i, d = res
        if isinstance(d, dict) and d:
            digests[i] = d
            ok += 1
    if ok:
        logger.info(f"[Digest] Succeeded for {ok}/{len(tasks)} entries (failed entries fall back to original trajectories)")
    else:
        logger.warning("[Digest] All digests failed; three-playbook revisions fall back to legacy full-trajectory rendering")
    return digests
