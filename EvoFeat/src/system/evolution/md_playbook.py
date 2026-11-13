from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..utils import get_logger


logger = get_logger("md_playbook")
CHANGELOG_HEADING = "Changelog"
CHANGELOG_HEADINGS = (CHANGELOG_HEADING,)

CHANGELOG_CAP = 50
_META_ORDER = ("kind", "version", "updated_at", "episodes")


READABILITY_CREED = """
Write for a reader unfamiliar with this system and graph algorithms. Explain technical terms in plain business English, follow observation -> action -> risk relevance, and provide complete executable details without generic claims or truncated sentences.
"""

_FIELD_RE = re.compile(r"^-\s+(?P<field>[^:]+):\s*(?P<value>.*)$")


def _split_card_head(s: str):
    s = (s or "").strip()
    if not s.startswith("### "):
        return None
    rest = s[4:].strip()
    if not rest:
        return None
    stats: dict = {}
    m = re.search(r"\s*\[([^\]]*)\]\s*$", rest)
    if m:
        for kv in m.group(1).split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                stats[k.strip()] = v.strip()
        rest = rest[:m.start()].rstrip()
    m = re.match(r"([^\s(\[]+)", rest)
    if not m:
        return None
    slug = m.group(1)
    tail = rest[m.end():].strip()
    title = ""
    if tail.startswith("("):
        title = tail[1:-1].strip() if tail.endswith(")") else tail[1:].strip()
    return slug, title, stats


@dataclass
class RuleCard:
    slug: str
    title: str = ""
    stats: dict = field(default_factory=dict)
    fields: list = field(default_factory=list)

    def field(self, name: str, default: str = "") -> str:
        for k, v in self.fields:
            if k == name:
                return v
        return default

    def set_field(self, name: str, value: str) -> None:
        value = " / ".join(str(value).splitlines())
        for i, (k, _) in enumerate(self.fields):
            if k == name:
                self.fields[i] = (name, value)
                return
        self.fields.append((name, value))


@dataclass
class Section:
    heading: str
    cards: list = field(default_factory=list)


@dataclass
class Playbook:
    meta: dict = field(default_factory=dict)
    title: str = ""
    sections: list = field(default_factory=list)
    changelog: list = field(default_factory=list)

    def iter_cards(self):
        for sec in self.sections:
            for card in sec.cards:
                yield sec, card

    def find_card(self, slug: str) -> RuleCard | None:
        for _, card in self.iter_cards():
            if card.slug == slug:
                return card
        return None

    def section_for(self, heading_prefix: str, create: bool = False,
                    heading: str | None = None) -> Section | None:
        for sec in self.sections:
            if sec.heading.startswith(heading_prefix):
                return sec
        if create:
            sec = Section(heading=heading or heading_prefix)
            self.sections.append(sec)
            return sec
        return None

    def add_changelog(self, entry: str) -> None:
        self.changelog.append(entry)
        del self.changelog[:-CHANGELOG_CAP]


def _normalize_episode_terms(pb: Playbook) -> None:
    if "episodes" not in pb.meta and "windows" in pb.meta:
        pb.meta["episodes"] = pb.meta["windows"]
    pb.meta.pop("windows", None)
    pb.meta.pop("used", None)
    pb.meta.pop("hit_rate", None)
    for _, card in pb.iter_cards():
        if "win" in card.stats:
            card.stats = {
                ("episode" if key == "win" else key): value
                for key, value in card.stats.items()
                if key != "win" or "episode" not in card.stats
            }
        card.stats.pop("used", None)
        if pb.meta.get("kind") == "synthesis_playbook":
            card.stats = {
                "revised": card.stats.get("revised", card.stats.get("updates", "0")),
                "last_rev": card.stats.get("last_rev", "0"),
                "episode": card.stats.get("episode", "0"),
            }


def parse_playbook(text: str, source: str = "<memory>") -> Playbook:
    pb = Playbook()
    lines = text.splitlines()
    i = 0

    if lines and lines[0].strip() == "---":
        i = 1
        while i < len(lines) and lines[i].strip() != "---":
            ln = lines[i]
            if ":" in ln:
                k, v = ln.split(":", 1)
                pb.meta[k.strip()] = v.strip()
            i += 1
        i += 1
    cur_section: Section | None = None
    cur_card: RuleCard | None = None
    in_changelog = False
    for ln in lines[i:]:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("# ") and not pb.title and cur_section is None and not in_changelog:
            pb.title = s[2:].strip()
            continue
        if s.startswith("## "):
            heading = s[3:].strip()
            in_changelog = (heading in CHANGELOG_HEADINGS)
            cur_card = None
            if not in_changelog:
                cur_section = Section(heading=heading)
                pb.sections.append(cur_section)
            continue
        if in_changelog:
            if s.startswith("- "):
                pb.changelog.append(s[2:].strip())
            else:
                logger.warning(f"[md_playbook] Skipped unrecognized changelog line ({source}): {ln[:60]}")
            continue
        if s.startswith("### "):
            parsed = _split_card_head(s)
            if not parsed:
                logger.warning(f"[md_playbook] Skipped unparseable entry heading ({source}): {ln[:60]}")
                cur_card = None
                continue
            if cur_section is None:
                cur_section = Section(heading="Entries")
                pb.sections.append(cur_section)
            slug, title, stats = parsed
            cur_card = RuleCard(slug=slug, title=title, stats=stats)
            cur_section.cards.append(cur_card)
            continue
        fm = _FIELD_RE.match(s)
        if fm and cur_card is not None:
            cur_card.fields.append((fm.group("field").strip(),
                                    fm.group("value").strip()))
            continue
        logger.warning(f"[md_playbook] Skipped unrecognized line ({source}): {ln[:60]}")
    _normalize_episode_terms(pb)
    return pb


def render_playbook(pb: Playbook) -> str:
    out = ["---"]
    keys = [k for k in _META_ORDER if k in pb.meta]
    keys += sorted(k for k in pb.meta if k not in _META_ORDER)
    for k in keys:
        out.append(f"{k}: {pb.meta[k]}")
    out.append("---")
    out.append("")
    out.append(f"# {pb.title or 'Playbook'}")
    for sec in pb.sections:
        out.append("")
        out.append(f"## {sec.heading}")
        for card in sec.cards:
            out.append("")
            head = f"### {card.slug}"
            if card.title:
                head += f" ({card.title})"
            if card.stats:
                stats_txt = ", ".join(f"{k}={v}" for k, v in card.stats.items())
                head += f" [{stats_txt}]"
            out.append(head)
            for fk, fv in card.fields:
                out.append(f"- {fk}: {fv}")
    out.append("")
    if pb.changelog:
        out.append(f"## {CHANGELOG_HEADING}")
        for entry in pb.changelog:
            out.append(f"- {entry}")
    out.append("")
    return "\n".join(out)


def prune_highlight_file(path: Path, marker: str) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        logger.warning(f"[md_playbook] Failed to read highlights file ({e}): {p}")
        return False
    keep: list[str] = []
    drop = False
    for ln in lines:
        if ln.startswith("## "):
            drop = marker in ln
        if not drop:
            keep.append(ln)
    while keep and not keep[-1].strip():
        keep.pop()
    text = "\n".join(keep).strip("\n") + "\n"
    if not text.strip() or not any(ln.startswith("## ") for ln in keep):
        try:
            p.unlink()
        except OSError as e:
            logger.warning(f"[md_playbook] Failed to delete highlights file ({e}): {p}")
        return True
    try:
        p.write_text(text, encoding="utf-8")
    except OSError as e:
        logger.warning(f"[md_playbook] Failed to write back highlights file ({e}): {p}")
        return False
    return True
