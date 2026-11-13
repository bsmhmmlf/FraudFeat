from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from ..llm.client import LLMClient
from ..paths import PLAYBOOK_DIR
from ..utils import ensure_dir, get_logger
from . import md_playbook as mdp


logger = get_logger("eval_skill_memory")
BOOK_ID = "eval_calibration"
BOOK_TITLE = "Evaluator Calibration Playbook"
SECTION_HEADING = "Family-level calibration rules"


HIGHLIGHT_MARKER = "evaluator-calibration"


_FIELD_ORDER = ("family", "distribution", "bias", "basis",
                "action", "avoid", "boundary")
_KEY_MAP = {"slug": "slug", "title": "title", "family": "family",
            "distribution": "distribution", "bias": "bias",
            "basis": "basis", "action": "action", "avoid": "avoid",
            "boundary": "boundary"}
_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SHORT_ID_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z]{1,3}\d{3}(?![0-9])")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{7,}$")
_REVISE_SYSTEM = """Revise the evaluator calibration playbook from current and historical IV-RAG entries. Learn class-level bias, not feature-specific instances.
Each rule has slug, title, family, distribution, bias, basis, action, avoid, and boundary. Require repeated same-family evidence before creating a rule; state evidence counts and concrete iv/conf adjustments.
Preserve unchanged rules by omission, remove only disproven rules, and keep one rule per family and bias pattern.
Return JSON only: {"rules":[{"slug":"","title":"","family":"","distribution":"","bias":"","basis":"","action":"","avoid":"","boundary":""}],"remove":[],"changelog":[]} or {"unchanged":true,"reason":""}.""" + mdp.READABILITY_CREED


_REWRITE_SYSTEM = """Rewrite supplied fields as class-level calibration guidance without concrete feature names, anonymous node IDs, or one-off instances. Preserve meaning, slug, and other fields. Return JSON only: {"cards":[{"slug":"","fields":{"field":"value"}}]}."""
_HIGHLIGHT_SYSTEM = """Summarize this episode's evaluator-calibration revisions in 2-4 clear English sentences, including the bias learned, evidence scale, adjustment, and expected impact. Return JSON only: {"summary":"..."}.""" + mdp.READABILITY_CREED


_INJECT_HEAD = "[EVALUATOR CALIBRATION PLAYBOOK — CLASS-LEVEL IV AND CONFIDENCE PRIORS]"


class EvalSkillMemory:
    def __init__(self, llm: LLMClient | None = None,
                 store_path: str | Path | None = None,
                 persist: bool = True, reset: bool = False,
                 read_only: bool = False):
        self.llm = llm or LLMClient()
        self.store_path = Path(store_path or (PLAYBOOK_DIR / f"{BOOK_ID}.md"))
        self.persist = bool(persist)
        self.read_only = bool(read_only)
        self._pb = self._empty_book()
        self._raw = ""
        self._load()
        if reset:
            self.clear()

    @staticmethod
    def _empty_book() -> mdp.Playbook:
        return mdp.Playbook(
            meta={"kind": "eval_calibration", "id": BOOK_ID,
                  "version": 0, "updated_at": "", "episodes": 0},
            title=BOOK_TITLE,
            sections=[mdp.Section(heading=SECTION_HEADING)],
            changelog=[])

    def _load(self) -> None:
        self._pb = self._empty_book()
        self._raw = ""
        if not (self.persist or self.read_only):
            return
        if not self.store_path.exists():
            return
        try:
            text = self.store_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning(f"Evaluator calibration playbook: read failed ({e}); continuing with an empty book: {self.store_path}")
            return
        if "### " not in text:
            self._raw = text.strip()
            logger.info(f"Evaluator calibration playbook: loaded prewritten prose -> {self.store_path}"
                        f" ({len(self._raw)} characters; injecting full text)")
            return
        pb = mdp.parse_playbook(text, source=str(self.store_path))
        if not pb.sections:
            pb.sections = self._empty_book().sections
        self._pb = pb
        logger.info(f"Evaluator calibration playbook: loaded {self.card_count()} calibration entries"
                    f" (version={pb.meta.get('version', '?')}) -> {self.store_path}")

    def save(self) -> None:
        if not self.persist:
            return
        if not self.card_count() and not self._pb.changelog:
            return
        ensure_dir(self.store_path.parent)
        self._pb.meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.store_path.write_text(mdp.render_playbook(self._pb), encoding="utf-8")
        logger.info(f"Evaluator calibration playbook: saved {self.card_count()} entries -> {self.store_path}")

    def clear(self) -> None:
        if not self.persist:
            logger.info("Evaluator calibration playbook: persist=False (frozen load); reset ignored, files retained")
            return
        self._pb = self._empty_book()
        self._raw = ""
        for f in (self.store_path,
                  self.store_path.parent / "_history" / f"{BOOK_ID}_snapshots.md"):
            if f.exists():
                try:
                    f.unlink()
                except OSError as e:
                    logger.warning(f"Evaluator calibration playbook: file deletion failed ({e}): {f}")
        mdp.prune_highlight_file(
            self.store_path.parent / "_history" / "evolution_highlights.md",
            HIGHLIGHT_MARKER)
        logger.info("Evaluator calibration playbook: cleared, including its snapshots and highlight sections")

    def card_count(self) -> int:
        return sum(1 for _ in self._pb.iter_cards())

    def rule_slugs(self) -> list:
        return sorted({c.slug for _, c in self._pb.iter_cards()})

    def playbook_text(self) -> str:
        if not self.card_count():
            return self._raw or ""
        return f"{_INJECT_HEAD}\n" + self._rules_text()

    def _rules_text(self) -> str:
        out: list[str] = []
        for card in (c for _, c in self._pb.iter_cards()):
            head = (f"### {card.title} (internal id: {card.slug})" if card.title
                    else f"### {card.slug}")
            out.append(head)
            out += [f"- {k}: {v}" for k, v in card.fields]
        return "\n".join(out) or "(no calibration rules yet)"

    def summarize(self, material: str, episode: int,
                  forbidden_names: set | None = None) -> dict | None:
        if not str(material or "").strip():
            logger.info("Evaluator calibration playbook: no material this episode; skipping revision (zero LLM calls)")
            return None
        self._pb.meta["episodes"] = int(self._pb.meta.get("episodes") or 0) + 1
        prompt = self._build_prompt(material)
        try:
            resp = self.llm.complete_json(_REVISE_SYSTEM, prompt, temperature=0.5)
        except Exception as e:
            logger.warning(f"Evaluator calibration playbook: revision call failed ({e}); preserving text this episode")
            return None
        if not isinstance(resp, dict) or not resp:
            logger.warning("Evaluator calibration playbook: no revision response or parsing failed; preserving text this episode")
            return None
        if resp.get("unchanged"):
            logger.info(f"Evaluator calibration playbook: no changes needed; preserving text"
                        f" ({str(resp.get('reason') or '').strip()})")
            return None
        remove_slugs = [str(x).strip() for x in (resp.get("remove") or []) if str(x).strip()]
        rules = self._parse_rules(resp)
        rules = self._lint_instances(rules, forbidden_names or set())
        rules = self._lint_family_dup(rules, remove_slugs)
        merged = self._merge(rules, remove_slugs)
        diff = self._apply(merged, episode)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        for entry in (resp.get("changelog") or []):
            entry = str(entry).strip()
            if entry:
                self._pb.add_changelog(f"[{ts} episode {episode}] {entry}")
        self.save()
        logger.info(f"Evaluator calibration playbook: episode {episode} revision completed: added {len(diff['added'])} / "
                    f"revised {len(diff['revised'])} / removed {len(diff['removed'])}")
        if diff["added"] or diff["revised"] or diff["removed"]:
            self._append_snapshot(episode, diff)
            self._write_highlight(episode, diff)
        return diff

    def seed_rules(self, rules: list[dict], *, source: str = "seed initialization",
                   note: str = "material: domain knowledge; no trajectories") -> dict:
        parsed = self._parse_rules({"rules": list(rules or [])})
        parsed = self._lint_instances(parsed, set())
        parsed = self._lint_family_dup(parsed, [])
        merged = self._merge(parsed, [])
        diff = self._apply(merged, 0)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._pb.add_changelog(f"[{ts} {source}] seed initialization: "
                               f"added {len(diff['added'])} rules ({note})")
        self.save()
        logger.info(f"Evaluator calibration playbook: seed injection completed: added {len(diff['added'])} / "
                    f"revised {len(diff['revised'])} / removed {len(diff['removed'])}")
        return diff

    def _build_prompt(self, material: str) -> str:
        lines = ["[EVALUATOR CALIBRATION PLAYBOOK: COMPLETE CURRENT RULES] "
                 "Return complete replacements for changed rules, list disproven rules in remove, and omit unchanged rules.",
                 self._rules_text() if self.card_count() else "(no calibration rules; create the first evidence-backed set)",
                 material]
        return "\n\n".join(lines)

    def _parse_rules(self, resp: dict) -> list[dict]:
        out: list[dict] = []
        bad: list[str] = []
        for r in (resp.get("rules") or []):
            if not isinstance(r, dict):
                continue
            d = {cn: str(r.get(en) or "").strip() for en, cn in _KEY_MAP.items()}
            slug = d.pop("slug", "")
            if not slug or not _SLUG_RE.match(slug):
                bad.append(slug or "(empty)")
                continue
            if not any(v for v in d.values()):
                continue
            title = d.pop("title", "")
            if not title and d.get("family"):
                title = d["family"][:16]
            out.append({"slug": slug, "title": title, **d})
        if bad:
            logger.warning(f"Evaluator calibration playbook: discarded {len(bad)} rules with invalid internal ids"
                           f" (English snake_case required): {bad}")
        return out

    def _lint_instances(self, rules: list[dict], forbidden: set) -> list[dict]:
        bad = [r for r in rules if self._instance_violation(r, forbidden)]
        if not bad:
            return rules
        logger.info(f"Evaluator calibration playbook: {len(bad)} rules contain instance-specific details/anonymous ids; requesting one rewrite")
        payload = json.dumps([{"slug": r["slug"], "fields": r} for r in bad],
                             ensure_ascii=False, indent=1)
        try:
            resp = self.llm.complete_json(_REWRITE_SYSTEM,
                                          f"Rewrite the following cards:\n{payload}", temperature=0.3)
        except Exception:
            resp = None
        rewritten: dict[str, dict] = {}
        for x in ((resp or {}).get("cards") or []):
            if isinstance(x, dict) and x.get("slug"):
                fields = x.get("fields") if isinstance(x.get("fields"), dict) else {}
                rewritten[str(x["slug"])] = {
                    _KEY_MAP.get(str(k), str(k)): str(v).strip()
                    for k, v in fields.items()}
        kept: list[dict] = []
        for r in rules:
            if r not in bad:
                kept.append(r)
                continue
            new_fields = rewritten.get(r["slug"]) or {}
            cand = {**r}
            for k, v in new_fields.items():
                if k in cand and v:
                    cand[k] = v
            if self._instance_violation(cand, forbidden):
                logger.warning(f"Evaluator calibration playbook: rule {r['slug']} remains instance-specific after rewriting; discarding entry")
                continue
            kept.append(cand)
        return kept

    @staticmethod
    def _instance_violation(rule: dict, forbidden: set) -> bool:
        text = " ".join(str(v) for k, v in rule.items()
                        if k not in ("slug", "title"))
        if _SHORT_ID_RE.search(text):
            return True
        for name in (forbidden or ()):
            n = str(name or "").strip()
            if n and _NAME_RE.match(n) and n in text:
                return True
        return False

    def _lint_family_dup(self, rules: list[dict], remove_slugs: list) -> list[dict]:
        rem = set(remove_slugs)
        seen: dict[str, str] = {}
        for _, card in self._pb.iter_cards():
            if card.slug in rem:
                continue
            key = _norm_key(card.field("family"))
            if key:
                seen.setdefault(key, card.slug)
        kept: list[dict] = []
        dropped: list[str] = []
        for r in rules:
            key = _norm_key(r.get("family"))
            if key and key in seen and seen[key] != r["slug"]:
                dropped.append(f"{r['slug']} (same family as {seen[key]})")
                continue
            if key:
                seen[key] = r["slug"]
            kept.append(r)
        if dropped:
            logger.warning(f"Evaluator calibration playbook: discarded {len(dropped)} duplicate-family rules"
                           f" (one entry per family; first arrival retained): {dropped}")
        return kept

    def _merge(self, rules: list[dict], remove_slugs: list) -> list[dict]:
        rem = set(remove_slugs)
        by_slug = {r["slug"]: r for r in rules}
        merged: list[dict] = []
        seen: set[str] = set()
        for _, card in self._pb.iter_cards():
            if card.slug in rem:
                logger.info(f"Evaluator calibration playbook: removed rule entry {card.slug} (disproven)")
                continue
            merged.append(by_slug.get(card.slug) or _card_to_rule(card))
            seen.add(card.slug)
        for r in rules:
            if r["slug"] not in seen:
                merged.append(r)
                seen.add(r["slug"])
        return merged

    def _apply(self, merged: list[dict], episode: int) -> dict:
        old = {c.slug: c for _, c in self._pb.iter_cards()}
        new_cards: list[mdp.RuleCard] = []
        added: list[str] = []
        revised: list[str] = []
        for r in merged:
            slug = r["slug"]
            prev = old.get(slug)
            card = mdp.RuleCard(slug=slug, title=str(r.get("title") or ""))
            if prev is None:
                card.stats.update({"revised": 0, "last_rev": episode,
                                   "episode": 1})
                added.append(slug)
            else:
                card.stats["revised"] = _as_int(prev.stats.get("revised"), 0)
                card.stats["last_rev"] = _as_int(prev.stats.get("last_rev"), 0)
                card.stats["episode"] = _as_int(prev.stats.get("episode"), 0) + 1
                if any(str(r.get(k) or "").strip() != prev.field(k)
                       for k in _FIELD_ORDER):
                    card.stats["revised"] += 1
                    card.stats["last_rev"] = episode
                    revised.append(slug)
            for k in _FIELD_ORDER:
                v = str(r.get(k) or "").strip()
                if v:
                    card.set_field(k, v)
            new_cards.append(card)
        removed = [s for s in old if s not in {c.slug for c in new_cards}]
        self._pb.sections[0].cards = new_cards
        self._pb.meta["version"] = _as_int(self._pb.meta.get("version"), 0) + 1
        return {"added": added, "revised": revised, "removed": removed}

    def _append_snapshot(self, episode: int, diff: dict) -> None:
        path = self.store_path.parent / "_history" / f"{BOOK_ID}_snapshots.md"
        try:
            ensure_dir(path.parent)
            if path.exists():
                m = re.search(r"last_snapshot_(?:episode|window):\s*(\d+)",
                              path.read_text(encoding="utf-8")[:400])
                if m and int(m.group(1)) >= episode:
                    return
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            block = (f"\n\n## Episode {episode} full set ({self.card_count()} rules)"
                     f" | added {len(diff['added'])} / revised {len(diff['revised'])} / "
                     f"removed {len(diff['removed'])} | {ts}\n\n" + self._rules_text())
            if not path.exists():
                path.write_text(
                    f"# {BOOK_TITLE} ({BOOK_ID}) full snapshots"
                    f" (appended when changed; injection always uses the current version)\n"
                    f"<!-- last_snapshot_episode: {episode} -->" + block + "\n",
                    encoding="utf-8")
            else:
                with path.open("a", encoding="utf-8") as out:
                    out.write(block + "\n")
                txt = re.sub(r"last_snapshot_episode:\s*\d+",
                             f"last_snapshot_episode: {episode}",
                             path.read_text(encoding="utf-8"), count=1)
                path.write_text(txt, encoding="utf-8")
            logger.info(f"Evaluator calibration playbook: appended full snapshot for episode {episode}")
        except OSError as e:
            logger.warning(f"Evaluator calibration playbook: full snapshot write failed ({e}): {path}")

    def _write_highlight(self, episode: int, diff: dict) -> None:
        path = self.store_path.parent / "_history" / "evolution_highlights.md"
        try:
            if path.exists() and f"## Episode {episode} | {HIGHLIGHT_MARKER} |" in \
                    path.read_text(encoding="utf-8"):
                return
        except OSError:
            pass

        def _names(slugs):
            out = []
            for s in slugs:
                card = self._pb.find_card(s)
                out.append(f"{card.title} (internal id: {s})" if card and card.title else s)
            return ", ".join(out) or "none"

        digest = (f"Episode {episode} evaluator-calibration revision digest: "
                  f"added=[{_names(diff['added'])}]; revised=[{_names(diff['revised'])}]; "
                  f"removed=[{', '.join(diff['removed']) or 'none'}]")
        try:
            resp = self.llm.complete_json(_HIGHLIGHT_SYSTEM, digest, temperature=0.4)
        except Exception as e:
            logger.warning(f"Evaluator calibration playbook: highlight summary call failed ({e}); skipping")
            return
        summary = str((resp or {}).get("summary") or "").strip()
        if not summary:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        try:
            ensure_dir(path.parent)
            if not path.exists():
                path.write_text("# Skill evolution highlights (per-episode LLM summaries)\n",
                                encoding="utf-8")
            with path.open("a", encoding="utf-8") as out:
                out.write(f"\n## Episode {episode} | {HIGHLIGHT_MARKER} | {ts}\n{summary}\n")
            self._pb.add_changelog(
                f"[{ts} episode {episode}] episode highlight: {summary.replace(chr(10), ' ').strip()}")
            self.save()
            logger.info(f"Evaluator calibration playbook: episode {episode} highlights written to {path.name}")
        except OSError as e:
            logger.warning(f"Evaluator calibration playbook: highlights write failed ({e}): {path}")


def _norm_key(v: str) -> str:
    txt = str(v or "")
    for ch in (" ", "\t", "\n"):
        txt = txt.replace(ch, "")
    return txt.lower()


def _card_to_rule(card: mdp.RuleCard) -> dict:
    d = {"slug": card.slug, "title": card.title or ""}
    for k, v in card.fields:
        d[k] = v
    return d


def _as_int(v, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default
