from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from ..llm.client import LLMClient
from ..parallel import run_parallel
from ..paths import MEMORY_DIR, PLAYBOOK_DIR
from ..utils import ensure_dir, get_logger
from ..core.schema import Skill
from . import md_playbook as mdp
from . import paradigm


logger = get_logger("skill_memory")
DEFAULT_SKILL_STORE = MEMORY_DIR / "skills.json"
DEFAULT_SKILL_MD_DIRNAME = PLAYBOOK_DIR.name


NAVIGATE_ID = "explore_navigate"
HYPOTHESIZE_ID = "feature_hypothesize"
PARADIGM_ID = "graph_paradigm"


BOOK_IDS = (PARADIGM_ID, NAVIGATE_ID, HYPOTHESIZE_ID)


EVAL_BOOK_PREFIX = "eval_calibration"


_STAGE_OF = {PARADIGM_ID: "paradigm", NAVIGATE_ID: "navigate",
             HYPOTHESIZE_ID: "hypothesize"}
_BOOK_OF_STAGE = {v: k for k, v in _STAGE_OF.items()}
BOOK_LABELS = {PARADIGM_ID: "Graph Paradigm Playbook", NAVIGATE_ID: "Navigation Playbook",
               HYPOTHESIZE_ID: "Feature Hypothesis Playbook"}
HIGHLIGHT_MARKER = "explorer-playbooks"
_FIELD_CAP = 500
_FIELD_CAP_PARADIGM = 600


def _field_cap(book_id: str) -> int:
    return _FIELD_CAP_PARADIGM if book_id == PARADIGM_ID else _FIELD_CAP


_METRIC_WORDS = re.compile(r"(?i)ground[- ]truth|mIV|marginal\s*IV|(?<![A-Za-z])IV(?![A-Za-z])|"
                           r"(positive|negative)\s+(trajector|sample)|hit\s+rate|accuracy|this\s+batch")


def _store_field(card: mdp.RuleCard, field: str, value: str,
                 cap: int, slug: str = "") -> None:
    value = str(value or "").strip()
    if not value:
        return
    if cap and len(value) > cap:
        logger.warning(f"Skill memory: {slug or card.slug}.{field} contains {len(value)} characters, "
                       f"exceeding the recommended limit {cap}; preserving full text without truncation"
                       f" (consider shortening or splitting into finer rules/paradigms next episode)")
    card.set_field(field, value)


_NAV_REVISE_SYSTEM = """Revise the navigation playbook for exploring partial graph structures before the full network is visible. Rules must rely on local observable signals and specify actions that complete the structure.
Each rule has slug, title, desc, signal, action, paradigms, example, risk, and boundary. Reference only paradigms supplied in the user message. Explain structural risk without metric evidence. Preserve unchanged rules by omission, revise existing slugs rather than duplicating them, and remove only disproven rules. Avoid concrete IDs, absolute amounts, and one-off instances.
Return JSON only: {"rules":[{"slug":"","title":"","desc":"","signal":"","action":"","paradigms":"","example":"","risk":"","boundary":""}],"remove":[],"changelog":[]} or {"unchanged":true,"reason":""}.""" + mdp.READABILITY_CREED
_HYP_REVISE_SYSTEM = """Revise the feature-hypothesis playbook for fully observed structures. Each rule links to one supplied paradigm and defines title, desc, paradigm, structure, feature_form, discrimination, example, risk, boundary, and a snake_case slug. Multiple rules may cover different metrics for one paradigm.
Explain normal versus fraud behavior and structural risk without metric evidence. Preserve unchanged rules, revise existing slugs rather than creating duplicates, and remove only disproven rules. Avoid concrete IDs, absolute amounts, and one-off instances.
Return JSON only: {"rules":[{"slug":"","title":"","desc":"","paradigm":"","structure":"","feature_form":"","discrimination":"","example":"","risk":"","boundary":""}],"remove":[],"changelog":[]} or {"unchanged":true,"reason":""}.""" + mdp.READABILITY_CREED


_REVISE_SYSTEM = {
    PARADIGM_ID: paradigm.PARADIGM_REVISE_SYSTEM + mdp.READABILITY_CREED,
    NAVIGATE_ID: _NAV_REVISE_SYSTEM,
    HYPOTHESIZE_ID: _HYP_REVISE_SYSTEM,
}


_REWRITE_SYSTEM = """Rewrite each supplied risk field as structure -> fraud pattern -> structural difference between fraud and normal accounts. Preserve all other fields and remove metric evidence. Return JSON only: {"rules":[{"slug":"","risk":""}]}."""
_HIGHLIGHT_SYSTEM = """Summarize this episode's revisions to the paradigm, navigation, and hypothesis playbooks for a research audience.
Return JSON only: {"overall":"3-5 sentences covering new knowledge, corrected assumptions, and impact","by_book":{"paradigm":"...","navigate":"...","hypothesize":"..."}}. Include only books with real changes in by_book. Use clear English and avoid generic claims.""" + mdp.READABILITY_CREED
_ENRICH_SYSTEM = """Fill only missing title and example fields for supplied exploration rules. Use a short English title and a sanitized 2-3 sentence example without concrete IDs or absolute amounts. Preserve all other fields and skip cards with no useful addition.
Return JSON only: {"cards":[{"book":"paradigm|navigate|hypothesize","slug":"","title":"","example":""}]}.""" + mdp.READABILITY_CREED


class SkillMemory:
    def __init__(self, llm: LLMClient | None = None,
                 store_path: str | Path | None = None,
                 persist: bool = True, reset: bool = False,
                 keep_slugs: set | None = None, read_only: bool = False):
        self.llm = llm or LLMClient()
        p = Path(store_path or DEFAULT_SKILL_STORE)
        if p.suffix.lower() == ".json":
            self.md_dir = p.parent / DEFAULT_SKILL_MD_DIRNAME
            self.legacy_json = p
        else:
            self.md_dir = p
            self.legacy_json = p / "skills.json"
        self.store_path = self.md_dir
        self.persist = persist
        self.read_only = bool(read_only)
        self._books: dict[str, mdp.Playbook] = {}
        self._raw_books: dict[str, str] = {}
        if reset:
            self._load()
            self.clear(keep_slugs)
        else:
            self._load()
            if persist:
                self._migrate_legacy()

    _BOOK_SKELETON = {
        PARADIGM_ID: ("Graph Paradigm Playbook", "Graph structure paradigms"),
        NAVIGATE_ID: ("Navigation Playbook", "Structural signals"),
        HYPOTHESIZE_ID: ("Feature Hypothesis Playbook", "Paradigm-to-feature mappings"),
    }

    @classmethod
    def _empty_book(cls, book_id: str) -> mdp.Playbook:
        title, heading = cls._BOOK_SKELETON.get(
            book_id, ("Feature Hypothesis Playbook", "Paradigm-to-feature mappings"))
        return mdp.Playbook(
            meta={"kind": "explorer_playbook", "id": book_id,
                  "stage": _STAGE_OF.get(book_id, "hypothesize"),
                  "version": 0, "updated_at": "", "episodes": 0},
            title=title,
            sections=[mdp.Section(heading=heading)],
            changelog=[])

    def _book_path(self, book_id: str) -> Path:
        return self.md_dir / f"{book_id}.md"

    def load(self) -> int:
        self._load()
        return sum(len(list(b.iter_cards())) for b in self._books.values())

    def _load(self) -> None:
        self._books = {}
        self._raw_books = {}
        can_read = bool(self.persist or self.read_only)
        parsed_cards: dict[str, int] = {}
        for bid in BOOK_IDS:
            path = self._book_path(bid)
            if not (can_read and path.exists()):
                self._books[bid] = self._empty_book(bid)
                continue
            text = path.read_text(encoding="utf-8")
            pb = mdp.parse_playbook(text, source=str(path))
            if not pb.sections:
                pb.sections = self._empty_book(bid).sections
            n_cards = sum(1 for _ in pb.iter_cards())
            parsed_cards[bid] = n_cards
            if n_cards == 0 and text.strip():
                self._raw_books[bid] = text.strip()
                logger.info(f"Skill memory: {BOOK_LABELS.get(bid, bid)} is prewritten prose; "
                            f"injecting all {len(self._raw_books[bid])} characters <- {path}")
            self._books[bid] = pb
            if n_cards:
                logger.info(f"Skill memory: loaded {BOOK_LABELS.get(bid, bid)} from {path}"
                            f" ({n_cards} rules, "
                            f"version={pb.meta.get('version', '?')})")
        self._sanitize_books()
        for bid in BOOK_IDS:
            if bid in self._raw_books or not parsed_cards.get(bid):
                continue
            if any(True for _ in self._books[bid].iter_cards()):
                continue
            text = self._book_path(bid).read_text(encoding="utf-8").strip()
            if text:
                self._raw_books[bid] = text
                logger.info(f"Skill memory: all parsed entries in {BOOK_LABELS.get(bid, bid)} were "
                            f"discarded by load checks -> injecting all {len(text)} characters as prewritten prose")
        self._raw_books = {b: v for b, v in self._raw_books.items()
                           if not any(True for _ in self._books[b].iter_cards())}

    def _sanitize_books(self) -> None:
        for bid, pb in self._books.items():
            self._migrate_field_names(bid, pb)
            self._dedup_cards(bid, pb)

    def _migrate_field_names(self, book_id: str, pb: mdp.Playbook) -> None:
        renamed = dropped = 0
        drops = set(self._FIELD_DROPS.get(book_id, ()))
        for _, card in pb.iter_cards():
            fields: list[tuple[str, str]] = []
            for k, v in card.fields:
                if k in drops:
                    dropped += 1
                    continue
                nk = self._FIELD_RENAMES.get(k, k)
                if nk != k:
                    renamed += 1
                if any(nk == ek for ek, _ in fields):
                    continue
                fields.append((nk, v))
            card.fields = fields
        if renamed or dropped:
            logger.info(f"Skill memory: {BOOK_LABELS.get(book_id, book_id)} field checks: "
                        f"renamed {renamed}, removed {dropped} obsolete fields")

    def _dedup_cards(self, book_id: str, pb: mdp.Playbook) -> None:
        bad_slug: list[str] = []
        kept_by_title: dict[str, mdp.RuleCard] = {}
        dup_dropped: list[str] = []
        for sec in pb.sections:
            cards: list[mdp.RuleCard] = []
            for card in sec.cards:
                if not _is_valid_slug(card.slug):
                    bad_slug.append(card.slug)
                    continue
                key = (card.title or "").strip()
                if not key:
                    cards.append(card)
                    continue
                prev = kept_by_title.get(key)
                if prev is None:
                    kept_by_title[key] = card
                    cards.append(card)
                    continue
                if _as_int(card.stats.get("last_rev"), 0) > \
                        _as_int(prev.stats.get("last_rev"), 0):
                    winner, loser = card, prev
                    cards.append(card)
                    for s in pb.sections:
                        if prev in s.cards:
                            s.cards.remove(prev)
                    if prev in cards:
                        cards.remove(prev)
                    kept_by_title[key] = card
                else:
                    winner, loser = prev, card
                winner.stats["episode"] = (_as_int(winner.stats.get("episode"), 0)
                                       + _as_int(loser.stats.get("episode"), 0))
                dup_dropped.append(f"{loser.slug} (same title as {winner.slug})")
            sec.cards = cards
        if bad_slug:
            logger.warning(f"Skill memory: {BOOK_LABELS.get(book_id, book_id)} discarded "
                           f"{len(bad_slug)} rules with invalid internal ids (non-English/empty ids cannot be tracked across episodes): "
                           f"{bad_slug}")
        if dup_dropped:
            logger.warning(f"Skill memory: {BOOK_LABELS.get(book_id, book_id)} merged "
                           f"{len(dup_dropped)} duplicate-title entries: {dup_dropped}")
        if bad_slug or dup_dropped:
            self._log_change(pb, f"load check: dropped {len(bad_slug)} rules with invalid ids, "
                                 f"merged {len(dup_dropped)} duplicate-title rules")

    def save(self) -> None:
        if not self.persist:
            return
        ensure_dir(self.md_dir)
        for bid, pb in self._books.items():
            pb.meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
            self._book_path(bid).write_text(mdp.render_playbook(pb), encoding="utf-8")
        n = sum(len(list(b.iter_cards())) for b in self._books.values())
        per_book = ", ".join(
            f"{BOOK_LABELS.get(b, b)} {sum(1 for _ in self._books[b].iter_cards())} entries"
            for b in BOOK_IDS if b in self._books)
        logger.info(f"Skill memory: saved three playbooks ({n} rules total: {per_book}) -> {self.md_dir}")

    def clear(self, keep_slugs: set | None = None) -> None:
        keep_slugs = set(keep_slugs or ())
        kept: list[tuple[str, mdp.RuleCard]] = []
        if keep_slugs:
            for bid, pb in self._books.items():
                for sec in pb.sections:
                    kept.extend((bid, c) for c in sec.cards
                                if c.slug in keep_slugs)
        self._books = {bid: self._empty_book(bid) for bid in BOOK_IDS}
        if self.persist:
            self._purge_history()
        if kept:
            for bid, card in kept:
                target = self._books[bid]
                target.sections[0].cards.append(card)
            if self.persist:
                if self.legacy_json.exists():
                    try:
                        self.legacy_json.unlink()
                    except OSError as e:
                        logger.warning(f"Skill memory: failed to delete legacy skills.json ({e})")
                self.save()
                logger.info(f"Skill memory: --reset retained {len(kept)} cold-start rule entries"
                            f" ({sorted(keep_slugs)}); organic artifacts cleared")
            return
        if self.persist:
            for f in [self._book_path(b) for b in BOOK_IDS] + [self.legacy_json]:
                if f.exists():
                    try:
                        f.unlink()
                    except OSError as e:
                        logger.warning(f"Skill memory: file deletion failed ({e}): {f}")
            if self.md_dir.exists():
                for f in self.md_dir.glob("*.md"):
                    if f.name.startswith("synthesis_playbook") \
                            or f.name.startswith(EVAL_BOOK_PREFIX):
                        continue
                    try:
                        f.unlink()
                    except OSError as e:
                        logger.warning(f"Skill memory: MD deletion failed ({e}): {f}")
        paradigm.set_active([])
        logger.info("Skill memory: cleared three playbooks, including legacy JSON/individual MD files; paradigm set unregistered")

    def _purge_history(self) -> None:
        hist = self.md_dir / "_history"
        if not hist.exists():
            return
        for bid in BOOK_IDS:
            p = hist / f"{bid}_snapshots.md"
            if p.exists():
                try:
                    p.unlink()
                    logger.info(f"Skill memory: deleted historical snapshot {p.name}")
                except OSError as e:
                    logger.warning(f"Skill memory: historical snapshot deletion failed ({e}): {p}")
        mdp.prune_highlight_file(hist / "evolution_highlights.md", HIGHLIGHT_MARKER)

    def _migrate_legacy(self) -> None:
        have_md = any(self._book_path(b).exists() for b in self._books)
        if have_md or not self.legacy_json.exists():
            return
        try:
            data = json.loads(self.legacy_json.read_text(encoding="utf-8"))
            legacy = [Skill.from_dict(d) for d in data.get("skills", [])]
        except Exception as e:
            logger.warning(f"Skill memory: failed to load legacy skills.json ({e}); continuing with an empty store")
            legacy = []
        if not legacy:
            try:
                self.legacy_json.unlink()
            except OSError:
                pass
            return
        logger.info(f"Skill memory: detected legacy library ({len(legacy)} skills); "
                    f"mechanically merging into navigation/hypothesis playbooks (excluding the paradigm playbook)")
        merged = self._mechanical_merge_legacy(legacy)
        for bid, rules in merged.items():
            self._apply_revision(self._books[bid], rules, episode=0,
                                 source="legacy merge")
        self._log_change(self._books[NAVIGATE_ID],
                         f"legacy migration: merged {len(legacy)} legacy skills into playbooks"
                         f" (skills.json and legacy individual MD files deleted)")
        self.save()
        try:
            self.legacy_json.unlink()
        except OSError as e:
            logger.warning(f"Skill memory: legacy JSON deletion failed ({e})")
        arch_md = self.md_dir.parent / "explorer_playbook_archive.md"
        for f in sorted(self.md_dir.glob("*.md")):
            if f.stem in BOOK_IDS or f.name.startswith("synthesis_playbook") \
                    or f.name.startswith(EVAL_BOOK_PREFIX):
                continue
            try:
                f.unlink()
            except OSError:
                pass
        arch_dir = self.md_dir / "_archive"
        if arch_dir.exists():
            for f in sorted(arch_dir.glob("*.md")):
                try:
                    with arch_md.open("a", encoding="utf-8") as out:
                        out.write(f.read_text(encoding="utf-8") + "\n\n")
                    f.unlink()
                except OSError:
                    pass
            try:
                arch_dir.rmdir()
            except OSError:
                pass

    @staticmethod
    def _mechanical_merge_legacy(legacy: list[Skill]) -> dict:
        out = {NAVIGATE_ID: [], HYPOTHESIZE_ID: []}
        for s in legacy:
            slug = re.sub(r"[^a-z0-9_]", "_", (s.anchor or s.id).lower()) or "legacy"
            if s.stage == "navigate":
                fields = {"slug": slug, "signal": s.when,
                          "action": " / ".join(s.procedure),
                          "risk": "(migrated legacy entry; risk to be filled from later trajectories)",
                          "boundary": ""}
                out[NAVIGATE_ID].append(fields)
            else:
                fields = {"slug": slug, "structure": s.when,
                          "feature_form": "",
                          "discrimination": " / ".join(s.procedure),
                          "risk": "(migrated legacy entry; risk to be filled from later trajectories)",
                          "boundary": ""}
                out[HYPOTHESIZE_ID].append(fields)
        return out

    @property
    def skills(self) -> list[Skill]:
        return [self._skill_of(bid) for bid in BOOK_IDS]

    def _skill_of(self, book_id: str) -> Skill:
        pb = self._books[book_id]
        cards = [c for _, c in pb.iter_cards()]
        s = Skill(id=book_id,
                  when=f"{pb.title} ({len(cards)} rules)",
                  procedure=[self._card_brief(c) for c in cards],
                  anchor=book_id, stage=pb.meta.get("stage", "hypothesize"))
        return s

    @staticmethod
    def _card_brief(card: mdp.RuleCard) -> str:
        head = (f"{card.title} (internal id: {card.slug})" if card.title else card.slug)
        first = card.fields[0] if card.fields else ("", "")
        return f"{head}: {first[0]}={first[1]}" if first[0] else head

    def list_all(self, stage: str | None = None) -> list[Skill]:
        bid = _BOOK_OF_STAGE.get(stage or "")
        if bid:
            return [self._skill_of(bid)]
        return self.skills

    def read_markdown(self, skill_id: str) -> str | None:
        pb = self._books.get(skill_id)
        if pb is None:
            for bid in BOOK_IDS:
                if skill_id.startswith(bid):
                    pb = self._books[bid]
                    break
        if pb is None:
            return None
        return mdp.render_playbook(pb)

    _INJECT_NOTE = {
        PARADIGM_ID: (
            "This playbook lists graph structure paradigms that recur in the network and carry risk semantics. "
            "Use substructures and clues to recognize the current path, risk to understand relevance, and boundary "
            "to distinguish adjacent paradigms. These paradigms are learned, not a mandatory checklist. The "
            "match_clues field is for deterministic feature classification and can be skipped when reading."),
        NAVIGATE_ID: (
            "Each navigation rule maps a locally observable signal to an action that completes a structure. "
            "Apply it only when its risk rationale fits the current evidence."),
        HYPOTHESIZE_ID: (
            "Each hypothesis rule maps a recognized paradigm to a statistic and discrimination mechanism. "
            "Choose the closest rule for the actual structure and ignore it when its risk rationale does not apply."),
    }

    def playbook_text(self, stage: str) -> str:
        bid = _BOOK_OF_STAGE.get(stage)
        if bid is None:
            logger.warning(f"Skill memory: unknown injection stage {stage!r}; returning an empty block (fallback)")
            return ""
        pb = self._books[bid]

        raw = self._raw_books.get(bid)
        if raw and not any(True for _ in pb.iter_cards()):
            logger.info(f"Skill memory: injecting full prose for {BOOK_LABELS.get(bid, bid)}"
                        f" ({len(raw)} characters)")
            return raw
        rules = self._rules_text(pb)
        note = self._INJECT_NOTE.get(bid, "")
        text = f"[{pb.title}] ({note})\n{rules}"
        if bid == PARADIGM_ID:
            n_cards = sum(1 for _ in pb.iter_cards())
            if n_cards:
                logger.info(f"Skill memory: paradigm playbook injection contains {n_cards} paradigms / "
                            f"about {len(text)} characters (present at every navigation hop; monitor prompt token cost "
                            f"as the number of entries grows)")
        return text

    def paradigm_specs(self) -> list:
        pb = self._books.get(PARADIGM_ID)
        if pb is None:
            return []
        return paradigm.parse_specs([c for _, c in pb.iter_cards()])

    def register_paradigms(self) -> int:
        return paradigm.set_active(self.paradigm_specs())

    def card_count(self, stage: str | None = None) -> int:
        bid = _BOOK_OF_STAGE.get(stage or "")
        bids = (bid,) if bid else BOOK_IDS
        return sum(len([1 for _ in self._books[b].iter_cards()]) for b in bids)

    def rule_slugs(self, stage: str | None = None) -> list[str]:
        bid = _BOOK_OF_STAGE.get(stage or "")
        bids = (bid,) if bid else BOOK_IDS
        return sorted({c.slug for b in bids for _, c in self._books[b].iter_cards()})

    def enrich_cards(self) -> int:
        todo: list[tuple[str, mdp.RuleCard]] = []
        for bid in BOOK_IDS:
            for _, card in self._books[bid].iter_cards():
                if not (card.title or "").strip() \
                        or not (card.field("example") or "").strip():
                    todo.append((bid, card))
        if not todo:
            logger.info("Skill memory: existing entry titles/examples are complete; no enrichment needed")
            return 0
        blocks = []
        for bid, card in todo:
            fields = "\n".join(f"  {k}: {v}" for k, v in card.fields)
            blocks.append(f"- book={_STAGE_OF.get(bid, bid)} slug={card.slug}\n{fields}")
        try:
            resp = self.llm.complete_json(
                _ENRICH_SYSTEM,
                "Fill the missing English title/example fields for these rule entries:\n" + "\n".join(blocks),
                temperature=0.4)
        except Exception as e:
            logger.warning(f"Skill memory: enrichment LLM call failed ({e}); skipping")
            return 0
        by_key = {}
        for x in ((resp or {}).get("cards") or []):
            if isinstance(x, dict):
                by_key[(str(x.get("book") or ""), str(x.get("slug") or ""))] = x
        filled = 0
        for bid, card in todo:
            x = by_key.get((_STAGE_OF.get(bid, bid), card.slug))
            if x is None:
                continue
            changed = False
            if not (card.title or "").strip() and str(x.get("title") or "").strip():
                card.title = str(x["title"]).strip()[:24]
                changed = True
            if not (card.field("example") or "").strip() \
                    and str(x.get("example") or "").strip():
                _store_field(card, "example", str(x["example"] or ""),
                             _field_cap(bid), card.slug)
                changed = True
            if changed:
                filled += 1
        if filled:
            self.save()
            logger.info(f"Skill memory: existing-entry enrichment completed ({filled}/{len(todo)} entries filled)")
        else:
            logger.info("Skill memory: enrichment LLM returned no usable additions; preserving original text")
        return filled

    def govern(self) -> dict:
        return {"retired": [], "merged": [], "merge_candidates": 0}

    def remove(self, ids: set) -> int:
        if not ids:
            return 0
        n = 0
        for pb in self._books.values():
            for sec in pb.sections:
                keep = [c for c in sec.cards
                        if c.slug not in ids and f"{c.slug}_v1" not in ids]
                n += len(sec.cards) - len(keep)
                sec.cards = keep
        if n:
            self.save()
            logger.info(f"Skill memory: removed {n} rule entries by id/slug (cold-start artifact cleanup)")
        return n

    def upsert(self, anchor: str, when: str, procedure: list,
               stage: str = "hypothesize") -> tuple:
        slug = re.sub(r"[^a-z0-9_]", "_", (anchor or "").strip().lower()) \
            or "failure_generic"
        pb = self._books[HYPOTHESIZE_ID]
        cur_episode = _as_int(pb.meta.get("episodes"), 0)
        card = pb.find_card(slug)
        fields = self._diagnosis_fields(when, procedure)
        if card is not None:
            changed = any(v and card.field(k) != v for k, v in fields.items())
            for k, v in fields.items():
                if v:
                    card.set_field(k, v)
            card.stats.pop("support", None)
            card.stats["episode"] = _as_int(card.stats.get("episode"), 1) + 1
            if changed:
                card.stats["revised"] = _as_int(card.stats.get("revised"), 0) + 1
                card.stats["last_rev"] = cur_episode
            self._log_change(pb, f"failure diagnosis reaffirmed {slug}: {(when or '')[:40]}")
            logger.info(f"Skill memory: reaffirmed rule entry {slug}")
            return self._skill_of(HYPOTHESIZE_ID), False
        sec = self._first_section(pb, HYPOTHESIZE_ID)
        card = mdp.RuleCard(slug=slug, stats={"revised": 0,
                                              "last_rev": cur_episode, "episode": 1})
        for k, v in fields.items():
            if v:
                card.set_field(k, v)
        sec.cards.append(card)
        self._log_change(pb, f"failure diagnosis added {slug}: {(when or '')[:40]}")
        logger.info(f"Skill memory: diagnosis added rule entry {slug} (hypothesis playbook)")
        return self._skill_of(HYPOTHESIZE_ID), True

    def add_rule(self, stage: str, slug: str, fields: dict, *,
                 source: str = "") -> tuple:
        bid = _BOOK_OF_STAGE.get(stage, HYPOTHESIZE_ID)
        slug = re.sub(r"[^a-z0-9_]", "_", (slug or "").strip().lower()) \
            or "rule_generic"
        pb = self._books[bid]
        cur_episode = _as_int(pb.meta.get("episodes"), 0)
        card = pb.find_card(slug)
        tag = source or "skill"
        cap = _field_cap(bid)
        fields = dict(fields or {})
        gist = (fields.get("structure") or fields.get("signal")
                or fields.get("structure") or fields.get("description") or "")[:40]
        if card is not None:
            changed = any(v and card.field(k) != v for k, v in fields.items())
            for k, v in fields.items():
                _store_field(card, k, v, cap, slug)
            card.stats.pop("support", None)
            card.stats["episode"] = _as_int(card.stats.get("episode"), 1) + 1
            if changed:
                card.stats["revised"] = _as_int(card.stats.get("revised"), 0) + 1
                card.stats["last_rev"] = cur_episode
            self._log_change(pb, f"{tag} updated {slug}: {gist}")
            logger.info(f"Skill memory: {tag} reaffirmed rule entry {slug} ({BOOK_LABELS.get(bid, bid)})")
            return self._skill_of(bid), False
        sec = self._first_section(pb, bid)
        card = mdp.RuleCard(slug=slug, stats={"revised": 0,
                                              "last_rev": cur_episode, "episode": 1})
        for k, v in fields.items():
            _store_field(card, k, v, cap, slug)
        sec.cards.append(card)
        self._log_change(pb, f"{tag} added {slug}: {gist}")
        logger.info(f"Skill memory: {tag} added rule entry {slug} ({BOOK_LABELS.get(bid, bid)})")
        return self._skill_of(bid), True

    @classmethod
    def _first_section(cls, pb: mdp.Playbook, book_id: str) -> mdp.Section:
        if pb.sections:
            return pb.sections[0]
        heading = cls._BOOK_SKELETON.get(
            book_id, ("", "Paradigm-to-feature mappings"))[1]
        return pb.section_for(heading, create=True, heading=heading)

    @staticmethod
    def _diagnosis_fields(when: str, procedure: list) -> dict:
        out = {"structure": when or "", "discrimination": "", "avoid": "",
               "risk": "", "boundary": ""}
        for step in (procedure or []):
            k, v = _split_step(step)
            if k == "do":
                out["discrimination"] = v
            elif k == "avoid":
                out["avoid"] = v
            elif k in ("risk", "boundary"):
                out[k] = v
        return out

    def summarize(self, episodes: list[dict], pool: list | None = None) -> list[Skill]:
        positives = [e for e in episodes if e.get("positive")]
        negatives = [e for e in episodes if e.get("negative")]
        if not positives and not negatives:
            logger.info("No positive or negative trajectories; skipping skill evolution (zero LLM calls for all three playbooks)")
            return []
        if any(e.get("forced") for e in episodes):
            logger.warning("Skill evolution batch includes forced fallback labels (sample quality may be low)")
        digests: list[dict | None] = []
        try:
            from .trajectory_digest import digest_trajectories
            digests = digest_trajectories(episodes, self.llm)
        except Exception as e:
            logger.warning(f"Skill summarization: feature-level digest layer failed ({e}); falling back to legacy full-trajectory rendering")
            digests = []
        digest_ok = any(digests)
        for pb in self._books.values():
            pb.meta["episodes"] = _as_int(pb.meta.get("episodes"), 0) + 1
        episode = _as_int(self._books[NAVIGATE_ID].meta.get("episodes"), 0)
        pool = list(pool or [])
        diffs: dict[str, dict] = {}
        changed: list[Skill] = []
        if self._pool_has_structure(pool):
            d = self._revise_book(PARADIGM_ID, episodes, pool, episode,
                                  digests=digests if digest_ok else None)
            if d:
                diffs[PARADIGM_ID] = d
                if d["added"] or d["revised"] or d["removed"]:
                    changed.append(self._skill_of(PARADIGM_ID))
        else:
            logger.info("Skill summarization: feature pool empty/no evidence chains; skipping paradigm revision this episode"
                        " (no invented paradigms; zero LLM calls)")
        self.register_paradigms()

        counts = paradigm.count_by_paradigm(pool)
        if self.retire_barren_paradigms(counts, episode):
            counts = paradigm.count_by_paradigm(pool)
            if PARADIGM_ID not in diffs:
                diffs[PARADIGM_ID] = {"added": [], "revised": [], "removed": []}
            if self._skill_of(PARADIGM_ID) not in changed:
                changed.append(self._skill_of(PARADIGM_ID))
        covered = self._paradigm_covered_slugs()
        nav_digest = paradigm.paradigm_digest(
            "navigate", self._paradigm_cards())
        hyp_digest = paradigm.paradigm_digest(
            "hypothesize", self._paradigm_cards(),
            counts=counts, covered_slugs=covered)
        ctx = {NAVIGATE_ID: nav_digest, HYPOTHESIZE_ID: hyp_digest}

        def _worker(bid: str, _index: int) -> tuple[str, dict | None]:
            return bid, self._revise_book(bid, episodes, pool, episode,
                                          extra_context=ctx.get(bid, ""),
                                          digests=digests if digest_ok else None)

        results = run_parallel([NAVIGATE_ID, HYPOTHESIZE_ID], _worker,
                               max_workers=2, stage="skill book revision")
        for item in results:
            if not item:
                continue
            bid, d = item
            if not d:
                continue
            diffs[bid] = d
            if d["added"] or d["revised"] or d["removed"]:
                changed.append(self._skill_of(bid))
        self.save()
        if changed:
            self._snapshot_and_highlight(episode, diffs, len(positives),
                                         len(negatives))
        logger.info(f"Skill summarization: episode {episode} revisions completed for all three playbooks; "
                    f"{len(changed)} books changed"
                    f" ({', '.join(BOOK_LABELS.get(s.id, s.id) for s in changed) or 'none'})")
        return changed

    @staticmethod
    def _pool_has_structure(pool: list) -> bool:
        for f in (pool or []):
            chain = (f.get("evidence_chain") if isinstance(f, dict)
                     else getattr(f, "evidence_chain", None))
            if chain:
                return True
        return False

    def _paradigm_cards(self) -> list:
        pb = self._books.get(PARADIGM_ID)
        return [c for _, c in pb.iter_cards()] if pb else []

    def _paradigm_covered_slugs(self) -> set:
        valid = paradigm.valid_slugs()
        covered: set[str] = set()
        pb = self._books.get(HYPOTHESIZE_ID)
        if pb is None:
            return covered
        for _, card in pb.iter_cards():
            for slug in _extract_slugs(card.field("paradigm"), valid):
                covered.add(slug)
        return covered

    def _revise_book(self, book_id: str, episodes: list[dict], pool: list,
                     episode: int, extra_context: str = "",
                     digests: list | None = None) -> dict | None:
        label = BOOK_LABELS.get(book_id, book_id)
        pb = self._books[book_id]
        prompt = self._build_book_prompt(book_id, episodes, pool, extra_context,
                                         digests=digests)
        try:
            resp = self.llm.complete_json(_REVISE_SYSTEM[book_id], prompt,
                                          temperature=0.6)
        except Exception as e:
            logger.warning(f"Skill summarization: {label} revision call failed ({e}); preserving text this round")
            return None
        if not isinstance(resp, dict) or not resp:
            logger.warning(f"Skill summarization: {label} no revision response/parsing failed; preserving text this round")
            return None
        if resp.get("unchanged"):
            logger.info(f"Skill summarization: {label} no changes needed; preserving text"
                        f" ({str(resp.get('reason') or '').strip()})")
            return None
        llm_rules = self._parse_rules(resp, book_id)
        llm_rules = self._lint_whys(llm_rules, book_id)
        llm_rules = self._lint_paradigm_refs(llm_rules, book_id)
        if book_id == PARADIGM_ID:
            llm_rules = self._lint_paradigm_clues(llm_rules)
        merged = self._merge_revision(book_id, llm_rules, resp.get("remove") or [])
        diff = self._apply_revision(pb, merged, episode=episode,
                                    source=f"episode {episode}")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        for entry in resp.get("changelog") or []:
            entry = str(entry).strip()
            if entry:
                pb.add_changelog(f"[{ts} episode {episode}] {entry}")
        logger.info(f"Skill summarization: {label} revision completed: added {len(diff['added'])} / "
                    f"revised {len(diff['revised'])} / removed {len(diff['removed'])}")
        return diff

    def seed_book(self, book_id: str, rules: list[dict], *,
                  source: str = "seed initialization",
                  note: str = "material: domain knowledge and schema vocabulary; no trajectories") -> dict:
        llm_rules = self._parse_rules({"rules": list(rules or [])}, book_id)
        llm_rules = self._lint_whys(llm_rules, book_id)
        llm_rules = self._lint_paradigm_refs(llm_rules, book_id)
        if book_id == PARADIGM_ID:
            llm_rules = self._lint_paradigm_clues(llm_rules)
        merged = self._merge_revision(book_id, llm_rules, [])
        diff = self._apply_revision(self._books[book_id], merged,
                                    episode=0, source=source)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._books[book_id].add_changelog(
            f"[{ts} {source}] seed initialization: added {len(diff['added'])} rules ({note})")
        logger.info(f"Skill summarization: {BOOK_LABELS.get(book_id, book_id)} seed injection completed: "
                    f"added {len(diff['added'])} / revised {len(diff['revised'])} / "
                    f"removed {len(diff['removed'])}")
        return diff

    def _build_book_prompt(self, book_id: str, episodes: list[dict],
                           pool: list, extra_context: str = "",
                           digests: list | None = None) -> str:
        label = BOOK_LABELS.get(book_id, book_id)
        lines = [f"[{label}: COMPLETE CURRENT RULES] Return complete replacements for changed rules, "
                 f"list disproven rules in remove, and omit unchanged rules.",
                 self._rules_text(self._books[book_id])]
        if book_id == PARADIGM_ID:
            lines.append(paradigm.split_digest(
                self._paradigm_cards(),
                counts=paradigm.count_by_paradigm(pool)))
            lines.append(self._pool_structure_text(pool))
        if extra_context:
            lines.append(extra_context)
        if digests:
            lines.append("[FEATURE TRAJECTORY DIGESTS: positive and negative sets derived from complete trajectories; "
                         "use these digests as the primary revision evidence]\n"
                         + self._digest_trajectory_text(episodes, digests))
        else:
            lines.append("[COMPLETE POSITIVE AND NEGATIVE EXPLORATION TRAJECTORIES: revision evidence]\n"
                         + self._trajectory_text(episodes))
        return "\n\n".join(lines)

    def _pool_structure_text(self, pool: list) -> str:
        from .family import get_field, rel_short, structure_signature
        rows: list[str] = []
        for f in (pool or []):
            chain = get_field(f, "evidence_chain", None) or []
            if not chain:
                continue
            name = (get_field(f, "feature_name_cn", "")
                    or get_field(f, "feature_name", "") or "(unnamed)")
            rels, terminal, evt = structure_signature(f)[:3]
            iv = get_field(f, "true_iv", None)
            try:
                iv_txt = f"{float(iv):.3f}" if iv is not None else "unmeasured"
            except (TypeError, ValueError):
                iv_txt = "unmeasured"
            cur = paradigm.paradigm_of(f)
            cur_txt = (paradigm.title_of(cur) if cur != paradigm.UNCLASSIFIED
                       else "unclassified")
            rows.append(f"- {name} | relation_path={'→'.join(rel_short(chain)) or '?'}"
                        f" | terminal={terminal or '?'} | value_type={evt or '?'}"
                        f" | discrimination={iv_txt} | paradigm={cur_txt}")
        if not rows:
            return "[CURRENT POOL STRUCTURE DISTRIBUTION] (no features with evidence chains)"
        head = (f"[CURRENT POOL STRUCTURE DISTRIBUTION: {len(rows)} features] Derive paradigms from these real structures. "
                f"If an unclassified structure recurs, prefer adding a paradigm. Broaden an existing match_clues only "
                f"when the feature genuinely instantiates the same template and keywords are incomplete; forcing heterogeneous "
                f"structures into one paradigm makes it too coarse.")
        return head + "\n" + "\n".join(rows)

    def _merge_revision(self, book_id: str, llm_rules: list[dict],
                        remove_slugs: list) -> list[dict]:
        pb = self._books[book_id]
        old = {c.slug: c for _, c in pb.iter_cards()}
        old_titles = {(c.title or "").strip(): c.slug
                      for _, c in pb.iter_cards() if (c.title or "").strip()}
        rem = set(str(x).strip() for x in (remove_slugs or []))
        by_slug = {r.get("slug"): r for r in llm_rules if r.get("slug")}
        merged: list[dict] = []
        seen: set[str] = set()
        for slug in old:
            if slug in rem:
                logger.info(f"Skill summarization: removed rule entry {slug} (disproven during revision)")
                continue
            merged.append(by_slug.get(slug) or self._card_to_rule(old[slug]))
            seen.add(slug)
        dup_titles: list[str] = []
        for r in llm_rules:
            s = r.get("slug")
            if not s or s in seen:
                continue
            title = str(r.get("title") or "").strip()
            owner = old_titles.get(title)
            if title and owner and owner not in rem:
                dup_titles.append(f"{s} -> merged into {owner} (title '{title}')")
                r["slug"] = owner
                for i, m in enumerate(merged):
                    if m.get("slug") == owner:
                        merged[i] = r
                        break
                continue
            merged.append(r)
            seen.add(s)
        if dup_titles:
            logger.warning(f"Skill summarization: {BOOK_LABELS.get(book_id, book_id)} "
                           f"merged {len(dup_titles)} new entries with existing titles as revisions: "
                           f"{dup_titles}")
        return merged

    @staticmethod
    def _card_to_rule(card: mdp.RuleCard) -> dict:
        d = {"slug": card.slug, "title": card.title or ""}
        for k, v in card.fields:
            d[k] = v
        return d

    def _apply_revision(self, pb: mdp.Playbook, rules: list[dict], *,
                        episode: int, source: str) -> dict:
        old = {c.slug: c for _, c in pb.iter_cards()}
        book_id = str(pb.meta.get("id", "") or "")
        sec = self._first_section(pb, book_id)
        cap = _field_cap(book_id)
        new_cards: list[mdp.RuleCard] = []
        seen: set[str] = set()
        added: list[str] = []
        revised: list[str] = []
        for r in rules:
            raw_slug = str(r.get("slug") or "").strip()
            slug = re.sub(r"[^a-z0-9_]", "_", raw_slug.lower())
            if not _is_valid_slug(slug):
                logger.warning(f"Skill summarization: skipping rule entry with invalid internal id '{raw_slug}'"
                               f" (English snake_case required)")
                continue
            if slug in seen:
                continue
            seen.add(slug)
            card = mdp.RuleCard(slug=slug, title=str(r.get("title") or ""))
            prev = old.get(slug)
            if prev is None:
                card.stats.update({"revised": 0,
                                   "last_rev": episode, "episode": 1})
                added.append(slug)
            else:
                card.stats["revised"] = _as_int(prev.stats.get("revised"), 0)
                card.stats["last_rev"] = _as_int(prev.stats.get("last_rev"), 0)
                card.stats["episode"] = _as_int(prev.stats.get("episode"), 0) + 1
            if prev is not None:
                fkeys = self._field_order(book_id)
                if (str(r.get("title") or "").strip() != (prev.title or "")
                        or any(str(r.get(k) or "").strip() != prev.field(k)
                               for k in fkeys)):
                    card.stats["revised"] += 1
                    card.stats["last_rev"] = episode
                    revised.append(slug)
            for k in self._field_order(book_id):
                _store_field(card, k, str(r.get(k) or ""), cap, slug)
            new_cards.append(card)
        removed = [slug for slug in old if slug not in seen]
        for slug in removed:
            self._log_change(pb, f"removed {slug} (absent from the {source} revision)")
        sec.cards = new_cards
        pb.meta["version"] = _as_int(pb.meta.get("version"), 0) + 1
        return {"added": added, "revised": revised, "removed": removed}

    _FIELD_ORDERS = {
        PARADIGM_ID: paradigm.PARADIGM_FIELD_ORDER,
        NAVIGATE_ID: ("description", "signal", "action", "paradigms",
                      "example", "risk", "boundary"),
        HYPOTHESIZE_ID: ("description", "paradigm", "structure", "feature_form",
                         "discrimination", "example", "risk", "boundary"),
    }

    @classmethod
    def _field_order(cls, book_id: str) -> tuple:
        return cls._FIELD_ORDERS.get(book_id, cls._FIELD_ORDERS[HYPOTHESIZE_ID])

    _KEY_MAPS = {
        PARADIGM_ID: {"slug": "slug", "title": "title", "desc": "description",
                      "structure": "structure", "substructures": "substructures",
                      "clues": "clues", "risk": "risk",
                      "boundary": "boundary", "match_clues": "match_clues"},
        NAVIGATE_ID: {"slug": "slug", "title": "title", "desc": "description",
                      "signal": "signal", "action": "action",
                      "paradigms": "paradigms", "example": "example",
                      "risk": "risk", "boundary": "boundary"},
        HYPOTHESIZE_ID: {"slug": "slug", "title": "title", "desc": "description",
                         "paradigm": "paradigm", "structure": "structure",
                         "feature_form": "feature_form",
                         "discrimination": "discrimination", "example": "example",
                         "risk": "risk", "boundary": "boundary"},
    }
    _FIELD_RENAMES = {"why": "risk"}
    _FIELD_DROPS = {PARADIGM_ID: ("example",)}

    def _parse_rules(self, node, book_id: str) -> list[dict]:
        if not isinstance(node, dict):
            return []
        key_map = self._KEY_MAPS.get(book_id, self._KEY_MAPS[HYPOTHESIZE_ID])
        out = []
        bad_slugs: list[str] = []
        for r in node.get("rules") or []:
            if not isinstance(r, dict):
                continue
            d = {cn: str(r.get(en) or "").strip() for en, cn in key_map.items()}
            if not d["slug"]:
                continue
            if not _is_valid_slug(d["slug"]):
                bad_slugs.append(d["slug"])
                continue
            if any(d[k] for k in key_map.values()
                   if k not in ("slug", "title")):
                if not d["title"] and d.get("description"):
                    d["title"] = d["description"][:16]
                out.append(d)
        if bad_slugs:
            logger.warning(f"Skill summarization: {BOOK_LABELS.get(book_id, book_id)} discarded "
                           f"{len(bad_slugs)} rules with invalid internal ids (English snake_case required): "
                           f"{bad_slugs}")
        return out

    def _lint_paradigm_clues(self, rules: list[dict]) -> list[dict]:
        pb = self._books[PARADIGM_ID]
        existing_slugs = {c.slug for _, c in pb.iter_cards()}

        seen_specs = list(paradigm.active())
        kept: list[dict] = []
        rejected: list[str] = []
        for r in rules:
            slug = str(r.get("slug") or "").strip()
            if not slug:
                continue
            spec = self._rule_to_spec(r)
            if slug in existing_slugs or spec is None:
                kept.append(r)
                if spec is not None:
                    seen_specs = [s for s in seen_specs if s.slug != slug] + [spec]
                continue
            clash = paradigm.indistinguishable_of(spec, seen_specs)
            if clash:
                rejected.append(f"{slug} (basis identical to {', '.join(clash[:3])})")
                continue
            kept.append(r)
            seen_specs.append(spec)
        if rejected:
            logger.warning(
                f"Skill summarization: graph paradigm playbook rejected {len(rejected)} new paradigms: "
                f"their machine basis (match_clues) is identical to existing paradigms and cannot capture any features: "
                f"{rejected}. Parent paradigms remain unchanged; next-episode checks will request a structured basis.")
            self._log_change(pb, f"rejected {len(rejected)} new paradigms with indistinguishable basis: "
                                 f"{'; '.join(rejected)}")
        return kept

    @staticmethod
    def _rule_to_spec(rule: dict):
        clue = str(rule.get("match_clues") or "").strip()
        if not clue:
            return None
        kws, terminal, vts = paradigm.parse_match_clues(clue)
        if not kws:
            return None
        (first_hop, has_ident, has_fund,
         has_op, has_obj) = paradigm.parse_edge_clues(clue)
        return paradigm.ParadigmSpec(
            slug=str(rule.get("slug") or "").strip(),
            title=str(rule.get("title") or "").strip(),
            keywords=kws, terminal=terminal, value_types=vts,
            first_hop=first_hop, has_identity=has_ident, has_fund=has_fund,
            has_operation=has_op, has_object=has_obj)

    BARREN_RETIRE_EPISODES = 3

    def retire_barren_paradigms(self, counts: dict, episode: int) -> list[str]:
        pb = self._books.get(PARADIGM_ID)
        if pb is None:
            return []
        retired: list[str] = []
        for sec in pb.sections:
            keep: list[mdp.RuleCard] = []
            for card in sec.cards:
                n = int((counts or {}).get(card.slug, 0) or 0)
                if n > 0:
                    if _as_int(card.stats.get("barren"), 0):
                        card.stats["barren"] = 0
                    keep.append(card)
                    continue
                if _as_int(card.stats.get("episode"), 1) <= 1:
                    keep.append(card)
                    continue
                barren = _as_int(card.stats.get("barren"), 0) + 1
                card.stats["barren"] = barren
                if barren >= self.BARREN_RETIRE_EPISODES:
                    retired.append(card.slug)
                    continue
                keep.append(card)
            sec.cards = keep
        if retired:
            logger.warning(
                f"Skill memory: graph paradigm playbook retired {len(retired)} paradigms with zero coverage for "
                f"{self.BARREN_RETIRE_EPISODES} consecutive episodes (capturing no features while "
                f"continuing to consume injection budget): {retired}")
            self._log_change(pb, f"retired {len(retired)} paradigms with zero coverage "
                                 f"for {self.BARREN_RETIRE_EPISODES} consecutive episodes: "
                                 f"{', '.join(retired)}")
            self.register_paradigms()
        return retired

    _PARADIGM_REF_FIELD = {NAVIGATE_ID: "paradigms", HYPOTHESIZE_ID: "paradigm"}

    def _lint_paradigm_refs(self, rules: list[dict], book_id: str) -> list[dict]:
        field = self._PARADIGM_REF_FIELD.get(book_id)
        if not field:
            return rules
        valid = paradigm.valid_slugs()
        if not valid:
            return rules
        dropped: list[str] = []
        for r in rules:
            raw = str(r.get(field) or "").strip()
            if not raw:
                continue
            hits = _extract_slugs(raw, valid)
            if hits:
                r[field] = "; ".join(paradigm.display_of(s) for s in hits)
            else:
                r[field] = ""
                dropped.append(f"{r.get('slug')}→{raw[:40]}")
        if dropped:
            logger.warning(f"Skill summarization: {BOOK_LABELS.get(book_id, book_id)} "
                           f"removed nonexistent paradigm references from {len(dropped)} rules"
                           f" (rule bodies retained): {dropped}")
        return rules

    RISK_FIELD = "risk"

    def _lint_whys(self, rules: list[dict], book_id: str) -> list[dict]:
        fld = self.RISK_FIELD
        bad = [r for r in rules if _METRIC_WORDS.search(r.get(fld, ""))]
        if not bad:
            return rules
        logger.info(f"Skill summarization: {len(bad)} rules contain metric terms in '{fld}'; requesting one rewrite")
        blocks = json.dumps([{"slug": r["slug"], "risk": r[fld]} for r in bad],
                            ensure_ascii=False, indent=1)
        try:
            resp = self.llm.complete_json(_REWRITE_SYSTEM,
                                          f"Rewrite the risk fields of these rules:\n{blocks}",
                                          temperature=0.3)
        except Exception:
            resp = None
        rewritten = {str(x.get("slug") or ""): str(x.get("risk") or "")
                     for x in ((resp or {}).get("rules") or []) if isinstance(x, dict)}
        kept: list[dict] = []
        bad_ids = {id(r) for r in bad}
        for r in rules:
            if id(r) not in bad_ids:
                kept.append(r)
                continue
            new_why = rewritten.get(r["slug"], "").strip()
            if new_why and not _METRIC_WORDS.search(new_why):
                r[fld] = new_why
                kept.append(r)
                self._log_change(self._books[book_id],
                                 f"revised {r['slug']}: risk contained metric words and was rewritten")
            else:
                self._log_change(self._books[book_id],
                                 f"dropped {r['slug']}: risk still contained metric words after rewrite")
                logger.warning(f"Skill summarization: rule {r['slug']} still contains metric terms after risk rewriting; discarding entry")
        return kept

    @staticmethod
    def _episode_head(e: dict) -> str:
        head = (f"- feature: {e.get('feature') or '?'}"
                + (f" ({e.get('feature_cn')})" if e.get("feature_cn") else "")
                + f" | iv={e.get('iv', e.get('miv'))}"
                + (f", reward={e.get('reward')}, redundancy={e.get('redundancy')}"
                   if e.get("reward") is not None else "")
                + (", forced_fallback" if e.get("forced") else "")
                + (", rescued_by_verification (previously pruned; measured IV confirms the label)"
                   if e.get("verified_pruned") else "")
                + (", unverified (pruned by low prediction without measurement; down-weight this weak signal)"
                   if e.get("unverified") else ""))
        if e.get("true_iv") is not None and e.get("predicted_iv") is not None:
            head += (f", calibration: predicted iv={e.get('predicted_iv')}"
                     f"/conf={e.get('predicted_conf')} -> measured iv={e.get('true_iv')}")
        return head

    @staticmethod
    def _episode_fmt(e: dict) -> str:
        trace = e.get("trace") or []
        steps: list[str] = []
        for i, t in enumerate(trace, 1):
            act = str(t.get("action", "") or "")
            obs = str(t.get("obs", "") or "")
            th = str(t.get("thought", "") or "")
            seg = f"{i}.{act}"
            if obs:
                seg += f" | observation: {obs}"
            if th:
                seg += f" | reason: {th}"
            steps.append(seg)
        tsum = "\n      ".join(steps) or "(no trajectory)"
        return (SkillMemory._episode_head(e) + f"\n    complete trajectory:\n      {tsum}")

    def _trajectory_text(self, episodes: list[dict]) -> str:
        pos = [e for e in episodes if e.get("positive")]
        neg = [e for e in episodes if e.get("negative")]
        lines: list[str] = []
        if pos:
            lines.append(f"[POSITIVE TRAJECTORIES: high value; {len(pos)} entries]")
            lines += [self._episode_fmt(e) for e in pos[:8]]
        if neg:
            lines.append(f"[NEGATIVE TRAJECTORIES: low value; {len(neg)} entries]")
            lines += [self._episode_fmt(e) for e in neg[:8]]
        return "\n".join(lines) or "(no trajectory samples)"

    _DIGEST_KEY_LABELS = (
        ("structure", "structural paradigm"),
        ("signals", "signal"),
        ("nav_path", "exploration path"),
        ("statistics", "statistical form"),
        ("risk", "risk semantics"),
        ("failure", "failure mechanism and pitfalls"),
        ("boundary", "boundary"),
        ("do", "DO"),
        ("avoid", "avoid"),
    )

    def _digest_trajectory_text(self, episodes: list[dict],
                                digests: list) -> str:
        by_index = {i: e for i, e in enumerate(episodes)
                    if e.get("positive") or e.get("negative")}
        pos_idx = sorted(i for i, e in by_index.items() if e.get("positive"))
        neg_idx = sorted(i for i, e in by_index.items() if e.get("negative"))

        def _fmt_card(i: int) -> str:
            e = by_index[i]
            d = digests[i] if 0 <= i < len(digests) else None
            lines = [self._episode_head(e)]
            if not isinstance(d, dict) or not d:
                lines.append("  WARNING: LLM digest failed; the complete trajectory follows as fallback:")
                lines.append(self._episode_fmt(e).split("\n", 1)[-1])
                return "\n".join(lines)
            for key, label in self._DIGEST_KEY_LABELS:
                val = str(d.get(key, "") or "").strip()
                if val:
                    block = ("\n" + "      ").join(val.splitlines())
                    lines.append(f"  {label}: {block}")
            return "\n".join(lines)

        out: list[str] = []
        if pos_idx:
            out.append(f"[POSITIVE TRAJECTORY DIGESTS: {len(pos_idx)} high-value entries derived from complete trajectories]")
            out += [_fmt_card(i) for i in pos_idx]
        if neg_idx:
            out.append(f"[NEGATIVE TRAJECTORY DIGESTS: {len(neg_idx)} low-value entries; failure mechanisms and pitfalls]")
            out += [_fmt_card(i) for i in neg_idx]
        return "\n".join(out) or "(no trajectory samples)"

    _REF_SHOW_MAX = 5
    _REF_FIELDS = ("paradigms", "paradigm")

    @classmethod
    def _compact_refs(cls, field: str, value: str) -> str:
        if field not in cls._REF_FIELDS:
            return value
        parts = [p.strip() for p in str(value or "").split(";") if p.strip()]
        if len(parts) <= cls._REF_SHOW_MAX:
            return value
        head = "; ".join(parts[:cls._REF_SHOW_MAX])
        return (f"{head}; ... {len(parts)} paradigms total "
                f"(many references indicate a generic rule; choose the closest match for the observed structure)")

    @classmethod
    def _rules_text(cls, pb: mdp.Playbook) -> str:
        lines: list[str] = []
        for sec in pb.sections:
            lines.append(f"## {sec.heading}")
            for card in sec.cards:
                head = (f"### {card.title} (internal id: {card.slug})"
                        if card.title else f"### {card.slug}")
                if card.stats:
                    head += (f" [carried={card.stats.get('episode', 1)} episodes"
                             f" | revisions={card.stats.get('revised', 0)}"
                             f" | last_revision={card.stats.get('last_rev', 0)}]")
                lines.append(head)
                for k, v in card.fields:
                    lines.append(f"- {k}: {cls._compact_refs(k, v)}")
        return "\n".join(lines) or "(no rules yet; derive the first set from these trajectories)"

    def _snapshot_and_highlight(self, episode: int,
                                diffs: dict[str, dict], n_pos: int,
                                n_neg: int) -> None:
        if not self.persist:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        for bid in BOOK_IDS:
            self._append_snapshot(bid, episode, diffs.get(bid) or {}, ts)
        self._write_highlight(episode, diffs, n_pos, n_neg, ts)

    def _append_snapshot(self, book_id: str, episode: int, diff: dict,
                         ts: str) -> None:
        path = self.md_dir / "_history" / f"{book_id}_snapshots.md"
        try:
            ensure_dir(path.parent)
            if path.exists():
                head_txt = path.read_text(encoding="utf-8")[:400]
                m = re.search(r"last_snapshot_(?:episode|window):\s*(\d+)", head_txt)
                if m and int(m.group(1)) >= episode:
                    return
            pb = self._books[book_id]
            n_cards = sum(1 for _ in pb.iter_cards())
            na, nr, nd = (len(diff.get("added") or []), len(diff.get("revised") or []),
                          len(diff.get("removed") or []))
            block = (f"\n\n## Episode {episode} full set ({n_cards} rules)"
                     f" | added {na} / revised {nr} / removed {nd} | {ts}\n\n"
                     + self._rules_text(pb))
            if not path.exists():
                path.write_text(
                    f"# {BOOK_LABELS.get(book_id, book_id)} ({book_id}) full snapshots"
                    f" (appended when changed; injection always uses the current version)\n"
                    f"<!-- last_snapshot_episode: {episode} -->"
                    + block + "\n", encoding="utf-8")
            else:
                with path.open("a", encoding="utf-8") as out:
                    out.write(block + "\n")
                txt = path.read_text(encoding="utf-8")
                txt = re.sub(r"last_snapshot_episode:\s*\d+",
                             f"last_snapshot_episode: {episode}", txt, count=1)
                path.write_text(txt, encoding="utf-8")
            logger.info(f"Skill memory: {BOOK_LABELS.get(book_id, book_id)} "
                        f"episode {episode} full snapshot appended "
                        f"({n_cards} entries, +{na}/~{nr}/-{nd})")
        except OSError as e:
            logger.warning(f"Skill memory: full snapshot write failed ({e}): {path}")

    def _write_highlight(self, episode: int, diffs: dict[str, dict],
                         n_pos: int, n_neg: int, ts: str) -> None:
        path = self.md_dir / "_history" / "evolution_highlights.md"
        try:
            if path.exists() and f"## Episode {episode} | {HIGHLIGHT_MARKER} |" in \
                    path.read_text(encoding="utf-8"):
                return
        except OSError:
            pass
        digest = [f"Episode {episode} revision digest (evidence: {n_pos} positive and {n_neg} negative trajectories):"]
        changed_books: list[str] = []
        for bid in BOOK_IDS:
            d = diffs.get(bid) or {}
            def _names(slugs, _bid=bid):
                out = []
                for s in slugs:
                    card = self._books[_bid].find_card(s)
                    out.append(f"{card.title} (internal id: {s})"
                               if card and card.title else s)
                return ", ".join(out) or "none"
            if (d.get("added") or d.get("revised") or d.get("removed")):
                changed_books.append(bid)
            digest.append(
                f"{BOOK_LABELS.get(bid, bid)} (book key: {_STAGE_OF.get(bid, bid)}): "
                f"added=[{_names(d.get('added') or [])}]; "
                f"revised=[{_names(d.get('revised') or [])}]; "
                f"removed=[{(', '.join(d.get('removed') or [])) or 'none'}]")
        if changed_books:
            digest.append("Write a book-specific focus only for these changed books: "
                          + ", ".join(_STAGE_OF.get(b, b) for b in changed_books))
        try:
            resp = self.llm.complete_json(_HIGHLIGHT_SYSTEM, "\n".join(digest),
                                          temperature=0.4)
        except Exception as e:
            logger.warning(f"Skill memory: highlight summary LLM call failed ({e}); skipping")
            return
        resp = resp if isinstance(resp, dict) else {}
        overall = str(resp.get("overall") or resp.get("highlight") or "").strip()
        by_book_raw = resp.get("by_book") or {}
        by_book: dict[str, str] = {}
        if isinstance(by_book_raw, dict):
            for bid in BOOK_IDS:
                stage = _STAGE_OF.get(bid, bid)
                txt = by_book_raw.get(stage) or by_book_raw.get(bid) or ""
                txt = str(txt).strip()
                if txt:
                    by_book[bid] = txt
        if not overall and not by_book:
            return
        try:
            ensure_dir(path.parent)
            if not path.exists():
                path.write_text("# Skill evolution highlights (per-episode LLM summaries)\n",
                                encoding="utf-8")
            with path.open("a", encoding="utf-8") as out:
                out.write(f"\n## Episode {episode} | {HIGHLIGHT_MARKER} | {ts}\n")
                if overall:
                    out.write(f"{overall}\n")
                for bid in BOOK_IDS:
                    seg = by_book.get(bid)
                    if seg:
                        out.write(f"\n**{BOOK_LABELS.get(bid, bid)} focus**: "
                                  f"{seg}\n")
            for bid in BOOK_IDS:
                seg = by_book.get(bid) or overall
                if not seg:
                    continue
                tag = "book focus" if by_book.get(bid) else "episode highlight"
                self._log_change(self._books[bid],
                                 f"{tag}: {str(seg).replace(chr(10), ' ').strip()}")
            logger.info(f"Skill memory: episode {episode} highlights written to {path.name} "
                        f"(book-specific focus for {len(by_book)}/{len(BOOK_IDS)} books)")
        except OSError as e:
            logger.warning(f"Skill memory: highlight write failed ({e}): {path}")

    @staticmethod
    def _log_change(pb: mdp.Playbook, entry: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        w = _as_int(pb.meta.get("episodes"), 0)
        pb.add_changelog(f"[{ts} episode {w}] {entry}")
        pb.meta["version"] = _as_int(pb.meta.get("version"), 0) + 1
        pb.meta["updated_at"] = datetime.now().isoformat(timespec="seconds")


def _extract_slugs(text: str, valid: set) -> list[str]:
    txt = str(text or "")
    if not txt or not valid:
        return []
    hits: list[str] = []
    for slug in sorted(valid, key=len, reverse=True):
        if slug in txt and slug not in hits:
            if not any(slug in h for h in hits):
                hits.append(slug)
    return hits


def _as_int(v, default=0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _is_valid_slug(slug) -> bool:
    s = str(slug or "").strip()
    return bool(_SLUG_RE.match(s))


def _as_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


_STEP_PREFIX = (
    (("do", "DO"), "do"),
    (("avoid", "AVOID"), "avoid"),
    (("risk", "why", "Why", "WHY"), "risk"),
    (("boundary",), "boundary"),
)


def _split_step(step: str) -> tuple[str, str]:
    txt = str(step).strip()
    for prefixes, canon in _STEP_PREFIX:
        for p in prefixes:
            if txt.startswith(p + ":"):
                return canon, txt[len(p) + 1:].strip()
    return "note", txt
