from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

from ..core.schema import Skill
from ..llm.client import LLMClient
from ..paths import MEMORY_DIR, PLAYBOOK_DIR
from . import md_playbook as mdp


logger = logging.getLogger(__name__)
DEFAULT_SYNTH_PLAYBOOK = PLAYBOOK_DIR / "synthesis_playbook.md"


_ANCHOR_RE = re.compile(r"^[a-z][a-z0-9_]*$")


ANCHOR_TAXONOMY = """depth_*: missing or excessive hops; relation_*: invented or incorrect rel values; aggregate_*: incompatible final reduction; query_form_*: malformed plan or placeholders; amount_edge_*: amount filtering or aggregation; exec_*: execution errors; consistency_*: semantic inconsistency."""


_FAMILY_HEADINGS = {
    "depth": "depth_* (hop coverage)",
    "relation": "relation_* (invented or misused relations)",
    "aggregate": "aggregate_* (aggregation and reduction)",
    "query_form": "query_form_* (output form)",
    "amount_edge": "amount_edge_* (amount-edge filters and aggregates)",
    "exec": "exec_* (execution errors)",
    "consistency": "consistency_* (consistency diagnosis)",
    "success": "success_* (general success patterns)",
}
_MISC_HEADING = "misc_* (other)"

HIGHLIGHT_MARKER = "synthesis-playbook"

_SYSTEM = """Learn reusable synthesis rules from two kinds of feature-to-Gremlin trajectories: terminal failures (repair exhaustion, execution failure, zero hit, semantic inconsistency) and repaired trajectories where an earlier plan was rejected but a later repair round passed all gates. Read the complete synthesis trace, including each generated plan, repair feedback, execution result, consistency diagnosis, and validation outcome, before identifying root causes.
For repaired trajectories, extract the transferable correction itself: which validation error fired, why the earlier plan was wrong, and what minimal edit made it pass. Treat the rejected plan and the accepted plan in the trace as a before/after pair.
Each rule has anchor, title, when, do, avoid, example, mechanism, and boundary. Make actions executable and explain the failure mechanism. Avoid concrete feature names, node IDs, and one-off instances; avoid must be supported by supplied evidence. Prefer updating existing anchors and do not merely restate deterministic validators.
New anchors must use these categories:
""" + ANCHOR_TAXONOMY + """
Return JSON only: {"skills":[{"anchor":"","title":"","when":"","do":"","avoid":"","example":"","mechanism":"","boundary":""}]}. Return {"skills":[],"reason":""} when there is no incremental lesson.""" + mdp.READABILITY_CREED


_HIGHLIGHT_SYSTEM = """Summarize this episode's synthesis-playbook revisions in 3-5 clear English sentences: new failure and repair knowledge, corrected assumptions, and expected impact on synthesis success. Return JSON only: {"highlight":"..."}.""" + mdp.READABILITY_CREED


_ENRICH_SYSTEM = """Fill only missing English title and example fields for supplied synthesis rules. Use sanitized transferable examples and preserve other fields. Return JSON only: {"skills":[{"anchor":"","title":"","example":""}]}.""" + mdp.READABILITY_CREED


TITLE_FIELD = "title"


MECHANISM_FIELD = "mechanism"

_STEP_PREFIX = (
    ((TITLE_FIELD,), TITLE_FIELD),
    (("do", "DO"), "do"),
    (("avoid", "AVOID"), "avoid"),
    (("example",), "example"),
    ((MECHANISM_FIELD, "why", "Why", "WHY"), MECHANISM_FIELD),
    (("boundary",), "boundary"),
)


def _step_to_field(step: str) -> tuple[str, str]:
    txt = str(step).strip()
    for prefixes, canon in _STEP_PREFIX:
        for p in prefixes:
            for sep in (":",):
                if txt.startswith(p + sep):
                    return canon, txt[len(p) + len(sep):].strip()
    return "note", txt


class SynthesisSkillMemory:
    def __init__(self, llm: LLMClient | None = None,
                 store_path: str | Path | None = None,
                 persist: bool = True, reset: bool = False):
        self.llm = llm or LLMClient()
        p = Path(store_path) if store_path else DEFAULT_SYNTH_PLAYBOOK
        if p.suffix.lower() == ".json":
            p = p.with_name("synthesis_playbook.md")
        self.store_path = p
        self.legacy_json = p.with_name("synthesis_skills.json")
        self.legacy_md_dir = p.parent / "synthesis_skills"
        self.archive_md = p.parent / "_archive" / "synthesis_playbook_archive.md"
        self.persist = persist
        self.skills: list[Skill] = []
        self._raw: str = ""
        self._meta: dict = {"kind": "synthesis_playbook", "version": 0,
                            "updated_at": "", "episodes": 0}
        self._changelog: list[str] = []
        if reset:
            self.clear()
        else:
            if store_path is None:
                self._migrate_old_default()
            self._migrate_legacy()
            self._load()

    @staticmethod
    def _merge_procedure(old: list[str], new: list[str]) -> list[str]:
        if not new:
            return list(old)
        if any(str(p).startswith("example") for p in new):
            return list(new)
        old_scene = next((p for p in old if str(p).startswith("example")), None)
        merged = list(new)
        if old_scene:
            idx = next((i for i, p in enumerate(merged)
                        if str(p).startswith((MECHANISM_FIELD, "why", "Why"))),
                       len(merged))
            merged.insert(idx, old_scene)
        return merged

    def list_all(self, stage: str | None = None) -> list[Skill]:
        if stage is None:
            return list(self.skills)
        return [s for s in self.skills if s.stage in (stage, "both")]

    def read_markdown(self, skill_id: str) -> str | None:
        for s in self.skills:
            if s.id == skill_id or s.anchor == skill_id:
                return self._card_markdown(s)
        return None

    def playbook_text(self) -> str:
        if not self.skills:
            return self._raw or ""
        head = ("[REUSABLE SYNTHESIS SKILLS: COMPLETE PLAYBOOKS] Apply applicable entries and declare their IDs in skills_used")
        return head + "\n\n" + "\n\n".join(self._card_markdown(s) for s in self.skills)

    def enrich_cards(self) -> int:
        def _has(s, prefix: str) -> bool:
            return any(str(p).startswith(prefix) for p in s.procedure)

        todo = [s for s in self.skills
                if not _has(s, "example") or not _has(s, TITLE_FIELD)]
        if not todo:
            logger.info("[Synthesis skills] Existing titles/examples are complete; no enrichment needed")
            return 0
        blocks = []
        for s in todo:
            blocks.append(f"- anchor={s.anchor}\n  when: {s.when}\n"
                          + "\n".join(f"  {p}" for p in s.procedure))
        try:
            resp = self.llm.complete_json(
                _ENRICH_SYSTEM,
                "Fill the missing English title/example fields for these synthesis skills:\n" + "\n".join(blocks),
                temperature=0.4)
        except Exception as e:
            logger.warning(f"[Synthesis skills] Enrichment LLM call failed ({e}); skipping")
            return 0
        by_anchor = {str(x.get("anchor") or ""): x
                     for x in ((resp or {}).get("skills") or []) if isinstance(x, dict)}
        filled = 0
        for s in todo:
            x = by_anchor.get(s.anchor or "")
            if not x:
                continue
            changed = False
            title = str(x.get("title") or "").strip()
            if title and not _has(s, TITLE_FIELD):
                s.procedure.insert(0, f"{TITLE_FIELD}: {title[:24]}")
                changed = True
            ex = str(x.get("example") or "").strip()
            if ex and not _has(s, "example"):
                if len(ex) > 500:
                    logger.warning(f"[Synthesis skills] {s.anchor or '?'} example contains {len(ex)} characters, "
                                   f"exceeding the recommended limit 500; preserving full text without truncation")
                s.procedure.append(f"example: {ex}")
                changed = True
            if changed:
                filled += 1
        if filled:
            self.save()
            logger.info(f"[Synthesis skills] Existing-entry enrichment completed ({filled}/{len(todo)} entries filled)")
        else:
            logger.info("[Synthesis skills] Enrichment LLM returned no usable additions; preserving original text")
        return filled

    def govern(self) -> dict:
        return {"retired": [], "merged": [], "merge_candidates": 0}

    def _archive(self, s: Skill, reason: str) -> None:
        if not self.persist:
            return
        try:
            self.archive_md.parent.mkdir(parents=True, exist_ok=True)
            with self.archive_md.open("a", encoding="utf-8") as f:
                f.write(self._card_markdown(s))
                f.write(f"\n> Retired from the active playbook ({datetime.now().isoformat(timespec='seconds')}): "
                        f"{reason}\n\n")
        except OSError as e:
            logger.warning(f"[Synthesis skills] Archiving failed ({e}): {s.id}")

    def upsert(self, anchor: str, when: str, procedure: list,
               stage: str = "synthesize") -> tuple:
        anchor = (anchor or "").strip() or "exec_generic"
        existing = next((s for s in self.skills if s.anchor == anchor), None)
        episode = int(self._meta.get("episodes", 0) or 0)
        if existing is not None:
            existing.when = when or existing.when
            existing.procedure = self._merge_procedure(existing.procedure,
                                                       list(procedure or []))
            existing.stats["revised"] = int(existing.stats.get("revised", 0)) + 1
            existing.stats["last_rev"] = episode
            existing.stats["episode"] = int(existing.stats.get("episode", 0)) + 1
            self._log_change(f"failure diagnosis reaffirmed {anchor} "
                             f"(revised={existing.stats['revised']}): when={existing.when[:40]}")
            logger.info(f"[Synthesis skills] Reaffirmed anchor={anchor} "
                        f"(revised={existing.stats['revised']})")
            return existing, False
        sid = f"{anchor}_v1"
        new = Skill(id=sid, when=when or "", procedure=list(procedure or []),
                    anchor=anchor, stage=stage)
        new.stats.update({"revised": 0, "last_rev": episode, "episode": 1})
        self.skills.append(new)
        self._log_change(f"failure diagnosis added {anchor}: when={(when or '')[:40]}")
        logger.info(f"[Synthesis skills] Diagnosis added {sid} (anchor={anchor})")
        return new, True

    def seed_skills(self, skills: list[dict], *,
                    source: str = "seed initialization",
                    note: str = "material: domain knowledge; no failure trajectories") -> list:
        out: list[Skill] = []
        bad: list[str] = []
        for sd in skills or []:
            if not isinstance(sd, dict):
                continue
            sk = self._normalize_skill(sd)
            anchor = str(sk.get("anchor") or "").strip()
            if not _ANCHOR_RE.match(anchor):
                bad.append(anchor or "(empty)")
                continue
            s, _is_new = self.upsert(anchor, sk.get("when") or "",
                                     sk.get("procedure") or [])
            out.append(s)
        if bad:
            logger.warning(f"[Synthesis skills] Seed injection discarded {len(bad)} invalid anchors"
                           f" (English snake_case required): {bad}")
        self._log_change(f"{source}: seeded {len(out)} synthesis skills ({note})")
        self.save()
        logger.info(f"[Synthesis skills] Seed injection completed: {len(out)} entries -> {self.store_path.name}")
        return out

    def summarize(self, episodes: list[dict]) -> list[Skill]:
        """Distil synthesis lessons from terminal failures and repaired successes.

        Terminal failures teach what breaks a query; repaired successes teach how a
        rejected plan was corrected, so intermediate failures are learnable too.
        """
        learnable = [e for e in episodes if e.get("negative") or e.get("repaired_pass")]
        if not learnable:
            logger.info("[Synthesis skills] No failed/repaired trajectories; skipping summarization")
            return []
        self._meta["episodes"] = int(self._meta.get("episodes", 0) or 0) + 1
        existing = sorted({s.anchor for s in self.skills if s.anchor})
        prompt = self._build_prompt(learnable, existing)
        resp = self.llm.complete_json(_SYSTEM, prompt, temperature=0.6)
        if isinstance(resp, dict) and isinstance(resp.get("skills"), list) \
                and not resp["skills"]:
            logger.info(f"[Synthesis skills] LLM found no incremental lessons in this batch; skipping learning this round"
                        f" (reason: {str(resp.get('reason', ''))[:80]})")
            self.save()
            return []
        llm_skills: list[dict] = []
        if resp and isinstance(resp, dict) and resp.get("skills"):
            for sd in resp["skills"]:
                if not isinstance(sd, dict):
                    continue
                llm_skills.append(self._normalize_skill(sd))
            logger.info(f"[Synthesis skills] LLM returned {len(llm_skills)} skills")
        if not llm_skills:
            if not self.skills:
                logger.warning("[Synthesis skills] LLM summarization failed and library is empty; using deterministic fallback skills")
                llm_skills = self._fallback_skills()
            else:
                logger.warning("[Synthesis skills] LLM summarization failed/returned invalid output; library is nonempty; skipping this round")
                self.save()
                return []
        created: list[Skill] = []
        added: list[str] = []
        revised: list[str] = []
        for sk in llm_skills:
            anchor = (sk["anchor"] or "").strip() or "exec_generic"
            existing_obj = next((s for s in self.skills if s.anchor == anchor), None)
            if existing_obj:
                old_when, old_proc = existing_obj.when, list(existing_obj.procedure)
                existing_obj.when = sk["when"] or existing_obj.when
                existing_obj.procedure = self._merge_procedure(
                    existing_obj.procedure, sk["procedure"])
                if (existing_obj.when, existing_obj.procedure) != (old_when, old_proc):
                    revised.append(anchor)
                existing_obj.stats["revised"] = int(
                    existing_obj.stats.get("revised", 0)) + 1
                existing_obj.stats["last_rev"] = int(
                    self._meta.get("episodes", 0) or 0)
                existing_obj.stats["episode"] = int(
                    existing_obj.stats.get("episode", 0)) + 1
                created.append(existing_obj)
                self._log_change(
                    f"revised {anchor} (revision {existing_obj.stats['revised']}): "
                    f"when={existing_obj.when[:40]}"
                    f" (evidence: {len(learnable)} failed/repaired trajectories)")
                logger.info(f"[Synthesis skills] Updated existing anchor={anchor}"
                            f" (revision {existing_obj.stats['revised']})")
            else:
                sid = f"{anchor}_v1"
                new = Skill(sid, sk["when"] or "", sk["procedure"], anchor,
                            stage="synthesize")
                new.stats.update({
                    "revised": 0,
                    "last_rev": int(self._meta.get("episodes", 0) or 0),
                    "episode": 1,
                })
                self.skills.append(new)
                created.append(new)
                added.append(anchor)
                self._log_change(
                    f"added {anchor}: when={new.when[:40]}"
                    f" (evidence: {len(learnable)} failed/repaired trajectories)")
                logger.info(f"[Synthesis skills] Added skill {sid} (anchor={anchor})")
        self.save()

        if added or revised:
            self._snapshot_and_highlight(added, revised, len(learnable))
        return created

    def _snapshot_and_highlight(self, added: list[str], revised: list[str],
                                n_neg: int) -> None:
        if not self.persist:
            return
        episode = int(self._meta.get("episodes", 0) or 0)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        hist = self.store_path.parent / "_history"
        path = hist / "synthesis_snapshots.md"
        try:
            hist.mkdir(parents=True, exist_ok=True)
            if path.exists():
                m = re.search(r"last_snapshot_(?:episode|window):\s*(\d+)",
                              path.read_text(encoding="utf-8")[:400])
                if m and int(m.group(1)) >= episode:
                    path = None
            if path is not None:
                block = (f"\n\n## Episode {episode} full set ({len(self.skills)} skills)"
                         f" | added {len(added)} / revised {len(revised)} | {ts}\n\n"
                         + "\n\n".join(self._card_markdown(s) for s in self.skills))
                if not path.exists():
                    path.write_text(
                        "# Synthesis playbook full snapshots (appended when changed; "
                        "injection always uses the current version)\n"
                        f"<!-- last_snapshot_episode: {episode} -->" + block + "\n",
                        encoding="utf-8")
                else:
                    with path.open("a", encoding="utf-8") as out:
                        out.write(block + "\n")
                    txt = path.read_text(encoding="utf-8")
                    txt = re.sub(r"last_snapshot_episode:\s*\d+",
                                 f"last_snapshot_episode: {episode}", txt, count=1)
                    path.write_text(txt, encoding="utf-8")
                logger.info(f"[Synthesis skills] Appended full snapshot for episode {episode}"
                            f" ({len(self.skills)} entries, +{len(added)}/~{len(revised)})")
        except OSError as e:
            logger.warning(f"[Synthesis skills] Full snapshot write failed ({e}): {path}")
        hp = hist / "evolution_highlights.md"
        try:
            if hp.exists() and f"## Episode {episode} | {HIGHLIGHT_MARKER} |" in \
                    hp.read_text(encoding="utf-8"):
                return
        except OSError:
            pass

        def _names(anchors) -> str:
            out = []
            for a in anchors:
                s = next((x for x in self.skills if x.anchor == a), None)
                title = ""
                if s is not None:
                    for p in s.procedure:
                        if str(p).startswith(TITLE_FIELD):
                            title = str(p).split(":", 1)[-1].strip()
                            break
                out.append(f"{title} (internal id: {a})" if title else a)
            return ", ".join(out) or "none"

        digest = [f"Episode {episode} revision digest (evidence: {n_neg} failed trajectories):",
                  f"Synthesis playbook: added=[{_names(added)}]; revised=[{_names(revised)}]"]
        try:
            resp = self.llm.complete_json(_HIGHLIGHT_SYSTEM, "\n".join(digest),
                                          temperature=0.4)
        except Exception as e:
            logger.warning(f"[Synthesis skills] Highlight summary LLM call failed ({e}); skipping")
            return
        text = str((resp or {}).get("highlight") or "").strip() \
            if isinstance(resp, dict) else ""
        if not text:
            return
        try:
            if not hp.exists():
                hp.write_text("# Skill evolution highlights (per-episode LLM summaries)\n",
                              encoding="utf-8")
            with hp.open("a", encoding="utf-8") as out:
                out.write(f"\n## Episode {episode} | {HIGHLIGHT_MARKER} | {ts}\n{text}\n")
            self._log_change(f"episode highlight: {str(text).replace(chr(10), ' ').strip()}")
            logger.info(f"[Synthesis skills] Episode {episode} revision highlights written to {hp.name}")
        except OSError as e:
            logger.warning(f"[Synthesis skills] Highlights write failed ({e}): {hp}")

    @staticmethod
    def _normalize_skill(sd: dict) -> dict:
        if any(k in sd for k in ("title", "do", "avoid", "example",
                                 "mechanism", "why", "boundary")):
            procedure: list[str] = []
            if str(sd.get("title") or "").strip():
                procedure.append(f"{TITLE_FIELD}: {str(sd['title']).strip()}")
            if str(sd.get("do") or "").strip():
                procedure.append(f"do: {str(sd['do']).strip()}")
            if str(sd.get("avoid") or "").strip():
                procedure.append(f"avoid: {str(sd['avoid']).strip()}")
            if str(sd.get("example") or "").strip():
                procedure.append(f"example: {str(sd['example']).strip()}")
            mech = str(sd.get("mechanism") or sd.get("why") or "").strip()
            if mech:
                procedure.append(f"{MECHANISM_FIELD}: {mech}")
            if str(sd.get("boundary") or "").strip():
                procedure.append(f"boundary: {str(sd['boundary']).strip()}")
            return {"anchor": sd.get("anchor", "") or "",
                    "when": sd.get("when", "") or "",
                    "procedure": procedure}
        return {"anchor": sd.get("anchor", "") or "",
                "when": sd.get("when", "") or "",
                "procedure": [str(p) for p in (sd.get("procedure") or [])
                              if str(p).strip()]}

    def _build_prompt(self, learnable: list[dict], existing_anchors: list[str]) -> str:
        def _fmt(e: dict) -> str:
            outcome = ("repaired_success" if e.get("repaired_pass")
                       else "terminal_failure")
            head = (f"- feature={e.get('feature')}, value_type={e.get('value_type')}, "
                    f"hops={e.get('hops')}, repairs={e.get('repairs')}, "
                    f"outcome={outcome}, fallback={e.get('fallback')}")
            parts = [head]
            if e.get("logic"):
                parts.append(f"  computation_logic: {str(e['logic'])[:200]}")
            if e.get("chain_view"):
                parts.append("  evidence_chain:\n" + "\n".join(
                    f"  {ln}" for ln in str(e["chain_view"]).splitlines()))
            errs = e.get("errors") or []
            if errs:
                parts.append("  validation_error_sequence:")
                parts += [f"    · {x}" for x in errs]
            hist = e.get("err_history") or []
            if hist:
                parts.append("  accumulated_failure_history_by_round:")
                parts += [f"    · {x}" for x in hist[:12]]
            if e.get("failed_plan"):
                fp = json.dumps(e["failed_plan"], ensure_ascii=False)
                parts.append(f"  last_failed_plan_truncated: {fp[:800]}")
            if e.get("exec_error"):
                parts.append(f"  execution_error: {e['exec_error']}")
            if e.get("exec_value") is not None:
                parts.append(f"  execution_value: {e['exec_value']!r}")
            if e.get("inconsistent"):
                parts.append(f"  consistency_diagnosis: {e.get('consistency', '')}")
            if e.get("explorer_fault"):
                parts.append("  routing: chain absent from the live graph (explorer issue, not query syntax)")
            if e.get("skills_used"):
                parts.append(f"  declared_skills: {e['skills_used']}")
            tr = e.get("synth_trace") or []
            if tr:
                rows = []
                for i, t in enumerate(tr, 1):
                    act = str(t.get("action") or "")
                    obs = str(t.get("obs") or "")
                    th = str(t.get("thought") or "")
                    seg = f"    {i}. {act}"
                    if th:
                        seg += f" ({th})"
                    if obs:
                        seg += f"\n       {obs}"
                    rows.append(seg)
                parts.append("  synthesizer_step_by_step_trajectory_verbatim_including_llm_repair_execution_consistency:\n"
                             + "\n".join(rows))
            return "\n".join(parts)

        anchors_txt = ("\n".join(f"  - {a}" for a in existing_anchors)
                       if existing_anchors else "  (empty; create as needed)")
        failed = [e for e in learnable if e.get("negative")]
        repaired = [e for e in learnable
                    if e.get("repaired_pass") and not e.get("negative")]
        blocks = []
        if failed:
            blocks.append(
                f"[TERMINAL FAILURES: repair exhaustion, execution failure, zero hit, or inconsistency] "
                f"({len(failed)} entries with validation errors, failure history, failed plans, and execution feedback):\n"
                + "\n".join(_fmt(e) for e in failed[:5]))
        if repaired:
            blocks.append(
                f"[REPAIRED TRAJECTORIES: an earlier plan was rejected, a later repair round passed every gate] "
                f"({len(repaired)} entries; the step-by-step trace pairs each rejected plan with the accepted plan):\n"
                + "\n".join(_fmt(e) for e in repaired[:5]))
        return (f"Existing anchors (reuse and update rather than renaming):\n{anchors_txt}\n\n"
                + "\n\n".join(blocks) + "\n\n"
                f"Derive synthesis skill entries. For terminal failures, explain the mechanism causing the deviation; "
                f"for repaired trajectories, derive the correction pattern (which error fired, why the earlier plan was wrong, "
                f"and which minimal edit fixed it). New anchors must use the declared category families and include all required fields. "
                f"Return empty skills when there is no incremental lesson. JSON only.")

    @staticmethod
    @staticmethod
    def _fallback_skills() -> list[dict]:
        logger.info("[synthesis skills] using three deterministic fallback entries")
        return [
            {"anchor": "depth_strict_fill",
             "when": "multi-hop feature with evidence_chain hops>=2",
             "procedure": [
                 "do: cover exactly the evidence_chain hops on the final main path; for terminal amount, aggregate the last evidence edge in that same hop without another endpoint traversal",
                 "avoid: stopping before the deepest signal or appending an extra aggregation traversal after the final evidence hop",
                 "mechanism: fewer hops discard the deep signal, while extra hops change the measured object and can silently produce a deeper feature",
                 "boundary: not applicable to one-hop statistics"]},
            {"anchor": "relation_grounding",
             "when": "filtering edges by relation",
             "procedure": [
                 "do: copy the exact rel value from each evidence_chain segment, write has('relation', rel), and follow the segment's stored direction",
                 "avoid: inventing relation names, using entity names as relation filter values, or guessing direction",
                 "mechanism: an invented relation does not exist in the graph and causes zero hits or execution errors",
                 "boundary: when no matching observed relation exists, reject the hypothesis rather than hard-coding a substitute"]},
            {"anchor": "aggregate_value_type_compat",
             "when": "selecting the final reduction",
             "procedure": [
                 "do: use count for count, exists for flag, scalar(sum/mean/max) or reduce for amount, and final.ratio with Python division for ratio",
                 "avoid: incompatible reductions such as amount sum for an expected count, or aggregation on the wrong edge",
                 "mechanism: incompatible reduction shifts semantics and corrupts downstream binning and discrimination measurement",
                 "boundary: summing amount is correct when the feature itself is an amount statistic"]},
        ]

    def _log_change(self, entry: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._changelog.append(f"[{ts}] {entry}")
        del self._changelog[:-mdp.CHANGELOG_CAP]
        self._meta["version"] = int(self._meta.get("version", 0) or 0) + 1
        self._meta["updated_at"] = datetime.now().isoformat(timespec="seconds")

    def save(self) -> None:
        if not self.persist:
            return
        if not self.skills and self._raw:
            return
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(self._render_md(), encoding="utf-8")
        logger.info(f"[Synthesis skills] Persisted {len(self.skills)} skills to {self.store_path}")

    def _load(self) -> None:
        self._raw = ""
        if not self.store_path.exists():
            logger.info(f"[Synthesis skills] Load: {self.store_path} does not exist; starting from scratch")
            return
        try:
            text = self.store_path.read_text(encoding="utf-8")
            pb = mdp.parse_playbook(text, source=str(self.store_path))
            skills = [self._card_to_skill(card) for _, card in pb.iter_cards()]
            if not skills and text.strip():
                self._raw = text.strip()
                logger.info(f"[Synthesis skills] {self.store_path.name} is prewritten prose; "
                            f"falling back to full-text injection of {len(self._raw)} characters")
            self.skills = skills
            self._meta.update(pb.meta)
            self._changelog = list(pb.changelog)
            logger.info(f"[Synthesis skills] Loaded {len(self.skills)} skills from {self.store_path}"
                        f" (version={self._meta.get('version', '?')})")
        except OSError as exc:
            logger.warning(f"[Synthesis skills] Load failed ({exc}); starting from scratch")
            self.skills = []

    def clear(self) -> None:
        self.skills = []
        self._changelog = []
        self._meta = {"kind": "synthesis_playbook", "version": 0,
                      "updated_at": "", "episodes": 0}
        if self.persist:
            self._purge_history()
            for p in (self.store_path, self.legacy_json):
                if p.exists():
                    try:
                        p.unlink()
                    except OSError as e:
                        logger.warning(f"[Synthesis skills] File deletion failed: {e}")
            if self.legacy_md_dir.exists():
                for q in sorted(self.legacy_md_dir.rglob("*.md")):
                    try:
                        q.unlink()
                    except OSError:
                        pass
        logger.info(f"[Synthesis skills] Skill library cleared (store={self.store_path})")

    def _purge_history(self) -> None:
        hist = self.store_path.parent / "_history"
        if not hist.exists():
            return
        p = hist / "synthesis_snapshots.md"
        if p.exists():
            try:
                p.unlink()
                logger.info(f"[Synthesis skills] Deleted historical snapshot {p.name}")
            except OSError as e:
                logger.warning(f"[Synthesis skills] Historical snapshot deletion failed ({e}): {p}")
        mdp.prune_highlight_file(hist / "evolution_highlights.md", HIGHLIGHT_MARKER)

    def _migrate_old_default(self) -> None:
        if not self.persist or self.store_path.exists():
            return
        old_md = MEMORY_DIR / "synthesis_playbook.md"
        if not old_md.exists():
            return
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_md), str(self.store_path))
            logger.info(f"[Synthesis skills] Detected old default location {old_md}; moved to {self.store_path}"
                        f" (same directory as the exploration playbooks)")
        except OSError as e:
            logger.warning(f"[Synthesis skills] Failed to move the old default playbook ({e}): {old_md}")
            return
        old_arch = MEMORY_DIR / "synthesis_playbook_archive.md"
        if old_arch.exists():
            try:
                self.archive_md.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old_arch), str(self.archive_md))
            except OSError as e:
                logger.warning(f"[Synthesis skills] Failed to move the legacy archive ({e}): {old_arch}")
        old_hist = MEMORY_DIR / "_history"
        if not old_hist.exists():
            return
        new_hist = self.store_path.parent / "_history"
        try:
            new_hist.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"[Synthesis skills] Failed to create new _history directory ({e}): {new_hist}")
            return
        for name in ("synthesis_snapshots.md",):
            src = old_hist / name
            if src.exists():
                try:
                    shutil.move(str(src), str(new_hist / name))
                except OSError as e:
                    logger.warning(f"[Synthesis skills] Failed to move legacy snapshot ({e}): {src}")
        hp = old_hist / "evolution_highlights.md"
        if hp.exists():
            text = hp.read_text(encoding="utf-8")
            blocks = [b for b in re.split(r"(?m)^(?=## )", text)
                      if f"| {HIGHLIGHT_MARKER} |" in b.splitlines()[0]] if text.strip() else []
            if blocks:
                dst = new_hist / "evolution_highlights.md"
                try:
                    if not dst.exists():
                        dst.write_text("# Skill evolution highlights (per-episode LLM summaries)\n",
                                       encoding="utf-8")
                    with dst.open("a", encoding="utf-8") as out:
                        for b in blocks:
                            out.write("\n" + b.strip() + "\n")
                except OSError as e:
                    logger.warning(f"[Synthesis skills] Failed to merge legacy highlights ({e}): {hp}")

    def _migrate_legacy(self) -> None:
        if not self.persist or self.store_path.exists() or not self.legacy_json.exists():
            return
        try:
            data = json.loads(self.legacy_json.read_text(encoding="utf-8"))
            self.skills = [Skill.from_dict(d) for d in data]
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.warning(f"[Synthesis skills] Failed to load legacy JSON for migration ({exc}); continuing with an empty library")
            self.skills = []
            return
        logger.info(f"[Synthesis skills] Detected legacy library {self.legacy_json}"
                    f" ({len(self.skills)} skills); migrating once to a single MD file")
        self._log_change(f"legacy migration: {len(self.skills)} skills moved from synthesis_skills.json "
                         f"into a single Markdown file")
        self.save()
        try:
            self.legacy_json.unlink()
        except OSError as e:
            logger.warning(f"[Synthesis skills] Legacy JSON deletion failed ({e}): {self.legacy_json}")
        if self.legacy_md_dir.exists():
            arch = self.legacy_md_dir / "_archive"
            if arch.exists():
                for q in sorted(arch.glob("*.md")):
                    try:
                        with self.archive_md.open("a", encoding="utf-8") as f:
                            f.write(q.read_text(encoding="utf-8") + "\n\n")
                    except OSError:
                        pass
            for q in sorted(self.legacy_md_dir.rglob("*.md")):
                try:
                    q.unlink()
                except OSError:
                    pass
            for d in sorted((p for p in self.legacy_md_dir.rglob("*") if p.is_dir()),
                            key=lambda p: len(p.parts), reverse=True):
                try:
                    d.rmdir()
                except OSError:
                    pass
            try:
                self.legacy_md_dir.rmdir()
            except OSError:
                pass
        logger.info("[Synthesis skills] Legacy migration completed: synthesis_skills.json and individual-MD directory deleted")

    def _render_md(self) -> str:
        pb = mdp.Playbook(meta=dict(self._meta), title="Synthesis Skill Playbook",
                          changelog=list(self._changelog))
        pb.meta["updated_at"] = pb.meta.get("updated_at") or \
            datetime.now().isoformat(timespec="seconds")
        grouped: dict[str, list[Skill]] = {}
        for s in self.skills:
            fam = next((name for name in _FAMILY_HEADINGS
                        if (s.anchor or "").startswith(name + "_")), "misc")
            grouped.setdefault(fam, []).append(s)
        for fam, heading in _FAMILY_HEADINGS.items():
            if grouped.get(fam):
                sec = pb.section_for(fam, create=True, heading=heading)
                sec.cards.extend(self._skill_to_card(s)
                                 for s in sorted(grouped[fam],
                                                 key=lambda x: x.anchor or x.id))
        if grouped.get("misc"):
            sec = pb.section_for("misc", create=True, heading=_MISC_HEADING)
            sec.cards.extend(self._skill_to_card(s)
                             for s in sorted(grouped["misc"],
                                             key=lambda x: x.anchor or x.id))
        return mdp.render_playbook(pb)

    @staticmethod
    def _skill_to_card(s: Skill) -> mdp.RuleCard:
        stats = s.stats or {}
        card = mdp.RuleCard(slug=s.anchor or s.id)
        card.stats = {
            "revised": int(stats.get("revised", 0)),
            "last_rev": int(stats.get("last_rev", 0) or 0),
            "episode": int(stats.get("episode", 0) or 0),
        }
        card.fields.append(("when", s.when))
        for step in s.procedure:
            k, v = _step_to_field(step)
            if k == TITLE_FIELD:
                card.title = v
                continue
            card.fields.append((k, v))
        return card

    @staticmethod
    def _card_to_skill(card: mdp.RuleCard) -> Skill:
        def _i(key, default=0):
            try:
                value = card.stats.get(key, default)
                return int(float(value)) if value not in (None, "") else default
            except (TypeError, ValueError):
                return default

        anchor = card.slug
        proc = ([f"{TITLE_FIELD}: {card.title}"] if (card.title or "").strip() else []) \
            + [f"{k}: {v}" for k, v in card.fields if k != "when"]
        s = Skill(id=f"{anchor}_v1", when=card.field("when"),
                  procedure=proc, anchor=anchor, stage="synthesize")
        s.stats.update({
            "revised": _i("revised", _i("updates")),
            "last_rev": _i("last_rev"),
            "episode": _i("episode"),
        })
        return s

    @staticmethod
    def _card_markdown(s: Skill) -> str:
        card = SynthesisSkillMemory._skill_to_card(s)
        stats_txt = ", ".join(f"{k}={v}" for k, v in card.stats.items())
        head = (f"### {card.title} (internal id: {card.slug})"
                if (card.title or "").strip() else f"### {card.slug}")
        lines = [f"{head} [{stats_txt}]"]
        lines += [f"- {k}: {v}" for k, v in card.fields]
        return "\n".join(lines)
