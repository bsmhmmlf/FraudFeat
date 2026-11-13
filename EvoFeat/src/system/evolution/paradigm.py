from __future__ import annotations

from dataclasses import dataclass

from ..utils import get_logger
from ..core.vocabulary import entity_category, public_relation, public_relation_endpoints
from .family import get_field, structure_signature


logger = get_logger("paradigm")
UNCLASSIFIED = "unclassified"


SATURATED_MIN = 10


SOFT_TIERS: tuple[tuple[int | None, str, str, str], ...] = (
    (1, "none", "uncovered",
     "No pool feature belongs to this paradigm. This is a high-priority exploration direction."),
    (4, "light", "light coverage",
     "This paradigm has light coverage and remains open for deeper exploration."),
    (7, "medium", "medium coverage",
     "Coverage is medium. New output must state a distinct observation angle: change the mediator, terminal entity, value type, or aggregation. Renaming an unchanged structure and caliber does not count."),
    (SATURATED_MIN, "strong", "near saturation",
     "This paradigm is near saturation. Unless there is a clearly new angle, spend the budget on less-covered paradigms."),
    (None, "hard", "saturated",
     "This paradigm is saturated. Do not rename or reskin the same feature; a variant counts as new only if it changes the relation sequence, terminal entity, value type, or aggregation."),
)
PARADIGM_FIELD_ORDER = (
    "description",
    "structure",
    "substructures",
    "clues",
    "risk",
    "boundary",
    "match_clues",
)
MACHINE_FIELDS = ("match_clues",)
SPLIT_HINT_RATIO = 0.15
SPLIT_HINT_MIN_N = 8


SPLIT_HINT_CHARS = 1500
_TERMINAL_PREFIXES = ("terminal=",)
_VALUE_PREFIXES = ("value=",)
_FIRST_HOP_PREFIXES = ("first_hop=",)
_HAS_IDENTITY_PREFIXES = ("has_identity=",)
_HAS_FUND_PREFIXES = ("has_fund=",)
_HAS_OPERATION_PREFIXES = ("has_operation=",)
_HAS_OBJECT_PREFIXES = ("has_object=",)
_HAS_EDGE_PREFIXES = (_HAS_IDENTITY_PREFIXES + _HAS_FUND_PREFIXES
                      + _HAS_OPERATION_PREFIXES + _HAS_OBJECT_PREFIXES)
_KIND_FUND = "fund"
_KIND_IDENTITY = "identity"
_KIND_OPERATION = "operation"
_KIND_OBJECT = "object"
_FIRST_HOP_ALIASES = {
    "fund": _KIND_FUND, "trade": _KIND_FUND,
    "payer": "payer", "payee": "payee",
    "identity": _KIND_IDENTITY, "operation": _KIND_OPERATION,
    "object": _KIND_OBJECT,
}
_TRUE_WORDS = {"true", "yes", "y", "1"}
_FALSE_WORDS = {"false", "no", "n", "0"}


_MIN_KEYWORD_LEN = 2


def edge_kind(rel: str) -> str:
    public = public_relation(rel)
    if not public:
        return "other"
    if public["is_trade"]:
        return _KIND_FUND
    categories = {entity_category(entity) for entity in public_relation_endpoints(rel)}
    if "Operation behavior (operation_behavior)" in categories:
        return _KIND_OPERATION
    if "Subject (subject)" in categories:
        return _KIND_IDENTITY
    if "Object (object)" in categories:
        return _KIND_OBJECT
    if "Account (account)" in categories:
        return "account"
    return "other"


def _kind_of_jf(jf: str) -> str:
    value = (jf or "").strip()
    if not value:
        return "other"
    kind = edge_kind(value)
    if kind == _KIND_FUND:
        low = value.lower()
        if low.endswith("|payer"):
            return "payer"
        if low.endswith("|payee"):
            return "payee"
    return kind


def _first_hop_kind(rels) -> str:
    if not rels:
        return ""
    head = str(rels[0] or "").strip()
    kind = _kind_of_jf(head) if "|" in head or "&" in head else edge_kind(head)
    if kind == _KIND_FUND:
        low = head.lower()
        if low.endswith("|payer") or low.endswith("payer"):
            return "payer"
        if low.endswith("|payee") or low.endswith("payee"):
            return "payee"
    return kind


def _first_hop_matches(declared: str, actual: str) -> bool:
    if not declared:
        return True
    if declared == actual:
        return True
    if declared == _KIND_FUND:
        return actual in ("payer", "payee", _KIND_FUND)
    return False


@dataclass(frozen=True)
class ParadigmSpec:
    slug: str
    title: str = ""
    keywords: tuple[str, ...] = ()
    terminal: str = ""
    value_types: tuple[str, ...] = ()
    first_hop: str = ""
    has_identity: bool | None = None
    has_fund: bool | None = None
    has_operation: bool | None = None
    has_object: bool | None = None

    @property
    def display(self) -> str:
        return f"{self.title} (internal id: {self.slug})" if self.title else self.slug

    @property
    def fingerprint(self) -> tuple:
        return (frozenset(self.keywords), self.terminal,
                frozenset(self.value_types), self.first_hop,
                self.has_identity, self.has_fund,
                self.has_operation, self.has_object)


def parse_match_clues(text: str) -> tuple[tuple[str, ...], str, tuple[str, ...]]:
    keywords: list[str] = []
    terminal = ""
    value_types: list[str] = []
    raw = str(text or "")
    for sep in (",", ";", "|", "/"):
        raw = raw.replace(sep, "\n")
    for item in raw.split("\n"):
        item = item.strip().strip("`\"'()")
        if not item:
            continue
        low = item.lower()
        hit_prefix = False
        for p in _TERMINAL_PREFIXES:
            if low.startswith(p):
                terminal = item[len(p):].strip().lower()
                hit_prefix = True
                break
        if hit_prefix:
            continue
        for p in _VALUE_PREFIXES:
            if low.startswith(p):
                for v in item[len(p):].split(" "):
                    v = v.strip().lower()
                    if v:
                        value_types.append(v)
                hit_prefix = True
                break
        if hit_prefix:
            continue
        if any(low.startswith(p) for p in
               (_FIRST_HOP_PREFIXES + _HAS_EDGE_PREFIXES)):
            continue
        if len(low) >= _MIN_KEYWORD_LEN and low not in keywords:
            keywords.append(low)
    return tuple(keywords), terminal, tuple(dict.fromkeys(value_types))


def parse_structural_clues(text: str) -> tuple[str, bool | None, bool | None]:
    first_hop, has_identity, has_fund, *_ = parse_edge_clues(text)
    return first_hop, has_identity, has_fund


def parse_edge_clues(text: str) -> tuple[str, bool | None, bool | None,
                                        bool | None, bool | None]:
    first_hop = ""
    has_identity: bool | None = None
    has_fund: bool | None = None
    has_operation: bool | None = None
    has_object: bool | None = None
    raw = str(text or "")
    for sep in (",", ";"):
        raw = raw.replace(sep, "\n")
    for item in raw.split("\n"):
        item = item.strip().strip("`\"'()")
        if not item:
            continue
        low = item.lower()
        for p in _FIRST_HOP_PREFIXES:
            if low.startswith(p):
                val = item[len(p):].strip().lower()
                first_hop = _FIRST_HOP_ALIASES.get(val, val)
                break
        for p in _HAS_IDENTITY_PREFIXES:
            if low.startswith(p):
                val = item[len(p):].strip().lower()
                if val in _TRUE_WORDS:
                    has_identity = True
                elif val in _FALSE_WORDS:
                    has_identity = False
                break
        for p in _HAS_FUND_PREFIXES:
            if low.startswith(p):
                val = item[len(p):].strip().lower()
                if val in _TRUE_WORDS:
                    has_fund = True
                elif val in _FALSE_WORDS:
                    has_fund = False
                break
        for p in _HAS_OPERATION_PREFIXES:
            if low.startswith(p):
                val = item[len(p):].strip().lower()
                if val in _TRUE_WORDS:
                    has_operation = True
                elif val in _FALSE_WORDS:
                    has_operation = False
                break
        for p in _HAS_OBJECT_PREFIXES:
            if low.startswith(p):
                val = item[len(p):].strip().lower()
                if val in _TRUE_WORDS:
                    has_object = True
                elif val in _FALSE_WORDS:
                    has_object = False
                break
    return first_hop, has_identity, has_fund, has_operation, has_object


def parse_specs(cards) -> list[ParadigmSpec]:
    specs: list[ParadigmSpec] = []
    skipped: list[str] = []
    for card in (cards or []):
        slug = str(getattr(card, "slug", "") or "").strip()
        if not slug:
            continue
        try:
            clue_text = card.field("match_clues")
        except Exception:
            clue_text = ""
        kws, terminal, vts = parse_match_clues(clue_text)
        (first_hop, has_ident, has_fund,
         has_op, has_obj) = parse_edge_clues(clue_text)
        if not kws:
            skipped.append(slug)
            continue
        specs.append(ParadigmSpec(slug=slug,
                                  title=str(getattr(card, "title", "") or "").strip(),
                                  keywords=kws, terminal=terminal, value_types=vts,
                                  first_hop=first_hop, has_identity=has_ident,
                                  has_fund=has_fund, has_operation=has_op,
                                  has_object=has_obj))
    if skipped:
        logger.warning(f"[paradigm] {len(skipped)} entries lack match_clues and cannot classify features: "
                       f"{sorted(skipped)} (keywords will be completed in the next episode)")
    return specs


def indistinguishable_of(spec: ParadigmSpec, others) -> list[str]:
    fp = spec.fingerprint
    return [s.slug for s in (others or [])
            if s.slug != spec.slug and s.fingerprint == fp]


_ACTIVE: list[ParadigmSpec] = []
_VERSION = 0
_CACHE: dict[tuple, str] = {}


def set_active(specs) -> int:
    global _ACTIVE, _VERSION, _CACHE
    _ACTIVE = [s for s in (specs or []) if isinstance(s, ParadigmSpec)]
    _VERSION += 1
    _CACHE = {}
    logger.info(f"[paradigm] Registered {len(_ACTIVE)} active paradigms (version {_VERSION}): "
                f"{[s.title or s.slug for s in _ACTIVE]}")
    return len(_ACTIVE)


def active() -> list[ParadigmSpec]:
    return list(_ACTIVE)


def valid_slugs() -> set[str]:
    return {s.slug for s in _ACTIVE}


def title_of(slug: str) -> str:
    for s in _ACTIVE:
        if s.slug == slug:
            return s.title or s.slug
    return slug


def display_of(slug: str) -> str:
    for s in _ACTIVE:
        if s.slug == slug:
            return s.display
    return slug


def _chain_edges(f) -> list[str]:
    out: list[str] = []
    for s in (get_field(f, "evidence_chain", None) or []):
        if not isinstance(s, dict):
            continue
        rel = str(s.get("rel") or s.get("rel_name") or "").strip()
        if rel:
            out.append(rel)
            continue
        legacy = str(s.get("join_field") or s.get("jf") or "").strip()
        if legacy:
            out.append(legacy)
    return out


def _match_text(f) -> tuple[str, tuple, list[str]]:
    sig = structure_signature(f)
    rels, terminal, evt = sig[:3]
    parts = list(rels) + [terminal, evt,
                          str(get_field(f, "feature_name", "") or ""),
                          str(get_field(f, "feature_name_cn", "") or "")]
    text = " ".join(p for p in parts if p).lower()
    kinds = [_kind_of_jf(e) for e in _chain_edges(f)]
    return text, sig, kinds


def _score(spec: ParadigmSpec, text: str, sig: tuple, kinds: list[str]) -> float:
    rels = sig[0] if sig else ()

    if spec.first_hop:
        first_kind = (kinds[0] if kinds else
                      (_first_hop_kind(rels) if rels else ""))
        if not _first_hop_matches(spec.first_hop, first_kind):
            return 0.0
    declared = [(spec.has_identity, _KIND_IDENTITY),
                (spec.has_fund, _KIND_FUND),
                (spec.has_operation, _KIND_OPERATION),
                (spec.has_object, _KIND_OBJECT)]
    if any(d is not None for d, _ in declared):
        kinds_set = {_fund_of(k) for k in (kinds or ())}
        for declared_val, kind in declared:
            if declared_val is not None and \
                    (kind in kinds_set) != declared_val:
                return 0.0
    score = sum(1.0 for kw in spec.keywords if kw in text)
    if score <= 0:
        return 0.0
    _, terminal, evt = sig[:3]
    if spec.terminal and terminal and spec.terminal == terminal.lower():
        score += 1.5
    if spec.value_types and evt and evt.lower() in spec.value_types:
        score += 0.5

    specificity = sum(1 for x in (spec.first_hop, spec.has_identity,
                                  spec.has_fund, spec.has_operation,
                                  spec.has_object)
                      if x not in ("", None))
    return score + 0.75 * specificity


def _fund_of(kind: str) -> str:
    return _KIND_FUND if kind in ("payer", "payee") else kind


def paradigm_of(f) -> str:
    if not _ACTIVE:
        return UNCLASSIFIED
    text, sig, kinds = _match_text(f)
    key = (sig, text, tuple(kinds), _VERSION)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    best_slug, best_score = UNCLASSIFIED, 0.0
    for spec in sorted(_ACTIVE, key=lambda s: s.slug):
        sc = _score(spec, text, sig, kinds)
        if sc > best_score:
            best_slug, best_score = spec.slug, sc
    _CACHE[key] = best_slug
    return best_slug


def same_paradigm(a, b) -> bool:
    pa = paradigm_of(a)
    if pa == UNCLASSIFIED:
        return False
    return pa == paradigm_of(b)


def count_by_paradigm(pool) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in (pool or []):
        if not (get_field(f, "evidence_chain", None) or []):
            continue
        slug = paradigm_of(f)
        counts[slug] = counts.get(slug, 0) + 1
    return counts


def coverage_tier(count: int) -> tuple[str, str, str]:
    n = max(0, int(count or 0))
    for upper, key, label, guidance in SOFT_TIERS:
        if upper is None or n < upper:
            return key, label, guidance
    last = SOFT_TIERS[-1]
    return last[1], last[2], last[3]


def is_saturated(count: int) -> bool:
    return coverage_tier(count)[0] == "hard"


def split_digest(cards=None, *, counts: dict | None = None) -> str:
    if not _ACTIVE:
        return ""
    counts = counts or {}
    by_slug = {}
    for card in (cards or []):
        slug = str(getattr(card, "slug", "") or "").strip()
        if slug:
            by_slug[slug] = card
    total = sum(int(v or 0) for v in counts.values()) or 0
    rows: list[str] = []
    overloaded: list[str] = []
    barren: list[str] = []
    groups: dict[tuple, list[str]] = {}
    for spec in _ACTIVE:
        groups.setdefault(spec.fingerprint, []).append(spec.display)
    dup_groups = [g for g in groups.values() if len(g) > 1]
    for spec in _ACTIVE:
        n = int(counts.get(spec.slug, 0) or 0)
        chars = 0
        card = by_slug.get(spec.slug)
        if card is not None:
            for name in PARADIGM_FIELD_ORDER:
                if name in MACHINE_FIELDS:
                    continue
                try:
                    chars += len(str(card.field(name) or ""))
                except Exception:
                    pass
        pct = (n * 100.0 / total) if total else 0.0
        flags = []
        if n >= SPLIT_HINT_MIN_N and total and (n / total) >= SPLIT_HINT_RATIO:
            flags.append(f"contains {pct:.0f}% of classified pool features "
                         f"(threshold {SPLIT_HINT_RATIO * 100:.0f}%)")
        if chars >= SPLIT_HINT_CHARS:
            flags.append(f"body has {chars} characters (threshold {SPLIT_HINT_CHARS})")
        tag = (" WARNING too coarse: " + "; ".join(flags)) if flags else ""
        decls = []
        if spec.first_hop:
            decls.append(f"first_hop={spec.first_hop}")
        for key, label in (("has_identity", "identity"), ("has_fund", "fund"),
                           ("has_operation", "operation"),
                           ("has_object", "object")):
            val = getattr(spec, key)
            if val is not None:
                decls.append(f"has_{label}_edge={str(val).lower()}")
        decl_txt = (" | structured_basis: " + ", ".join(decls)) if decls else \
            " | WARNING no structured basis; keywords alone cannot distinguish similar paradigms"
        zero_tag = ""
        if n == 0:
            zero_tag = " WARNING zero coverage: no feature assigned"
            barren.append(spec.display)
        rows.append(f"- {spec.display} | pool_features={n} ({pct:.0f}%)"
                    f" | body_chars={chars}{decl_txt}{tag}{zero_tag}")
        if flags:
            overloaded.append(spec.display)
    lines = [f"[PARADIGM GRANULARITY AND BASIS AUDIT: {len(_ACTIVE)} paradigms, {total} classified pool features]"]
    lines += rows

    if dup_groups:
        pairs = "; ".join(" / ".join(g) for g in dup_groups[:6])
        lines.append(
            f"[HIGHEST PRIORITY: INDISTINGUISHABLE BASES] These paradigms have identical machine bases (match_clues), "
            f"so deterministic assignment cannot distinguish them: {pairs}"
            + (" ..." if len(dup_groups) > 6 else "")
            + ". Do not add paradigms this episode. First give them mutually exclusive structured bases such as "
              "`first_hop=payer|payee|identity|operation|object` and "
              "`has_identity=true|false`, `has_fund=true|false`, "
              "`has_operation=true|false`, or `has_object=true|false`. "
              "If they cannot be made mutually exclusive, merge them and list redundant entries in remove.")
    if barren:
        lines.append(
            f"[ZERO-COVERAGE PARADIGMS] These entries receive no features because their bases are ineffective "
            f"or the structures do not exist: {', '.join(barren[:8])}"
            + (" ..." if len(barren) > 8 else "")
            + ". Either repair match_clues with keywords or structured bases that capture the real structure, "
              "or remove duplicate reskins. Do not leave zero-coverage entries consuming injection budget.")
    if overloaded:
        lines.append(
            "[OVERLY COARSE PARADIGMS] Warned entries cover too many features or have excessively long bodies. "
            "Split only when the children have mutually exclusive structured bases, for example first_hop=payee "
            "versus first_hop=payer or has_identity=false versus true. Otherwise children receive no features. "
            "Use real structural differences: fund direction, identity/operation/object edges, or terminal "
            "wechat_pay_account, merchant_pay_account, bank_card, phone_number, "
            "wechat_pay_bill_deletion_event, or wechat_pay_card_back_information_query_event. "
            "Keep the original slug for the closest child.")
    if not (dup_groups or barren or overloaded):
        lines.append("[STATUS] Paradigm granularity and bases are healthy. Add a paradigm with a structured basis "
                     "when the pool contains uncovered structures; broaden match_clues only for wording differences "
                     "within the same structural template.")
    return "\n".join(lines)


def paradigm_digest(purpose: str, cards=None, *, counts: dict | None = None,
                    covered_slugs=None) -> str:
    if not _ACTIVE:
        return ""
    by_slug = {}
    for card in (cards or []):
        slug = str(getattr(card, "slug", "") or "").strip()
        if slug:
            by_slug[slug] = card

    def _fld(slug: str, name: str) -> str:
        card = by_slug.get(slug)
        if card is None:
            return ""
        try:
            return str(card.field(name) or "").strip()
        except Exception:
            return ""

    counts = counts or {}
    covered = set(covered_slugs or ())
    lines: list[str] = []
    if purpose == "navigate":
        lines.append("[CURRENT GRAPH PARADIGMS: navigation rules must reference these paradigms and substructures]")
        for spec in _ACTIVE:
            lines.append(f"- {spec.display}")
            for name in ("structure", "substructures", "clues"):
                v = _fld(spec.slug, name)
                if v:
                    lines.append(f"  {name}: {v}")
        lines.append("Navigation sees only the current-hop neighborhood, not the complete subgraph. "
                     "Write rules for locally recognizable substructures and prefer reusable signals that serve multiple paradigms.")
    else:
        lines.append("[CURRENT GRAPH PARADIGMS: every hypothesis rule must reference one]")
        gaps: list[str] = []
        for spec in _ACTIVE:
            n = int(counts.get(spec.slug, 0) or 0)
            _, label, _ = coverage_tier(n)
            has_card = spec.slug in covered
            lines.append(f"- {spec.display} | pool_features={n} ({label}) "
                         f"| {'hypothesis_rule_present' if has_card else 'missing_hypothesis_rule'}")
            for name in ("structure", "substructures"):
                v = _fld(spec.slug, name)
                if v:
                    lines.append(f"  {name}: {v}")
            if not has_card:
                gaps.append(spec.display)
        if gaps:
            lines.append(f"[GAPS] Prioritize hypothesis rules for: {', '.join(gaps)}")
        lines.append("One paradigm may have several hypothesis rules for distinct structural emphases or statistics. "
                     "Use the trajectories to decide which angles merit separate entries; do not pad or limit yourself to existing angles.")
    return "\n".join(lines)


PARADIGM_REVISE_SYSTEM = """Revise the graph-structure paradigm playbook using current pool coverage and positive/negative trajectories. A paradigm is a transferable risk-relevant structure, not a one-off graph.
Each rule has slug, title, desc, structure, substructures, clues, risk, boundary, and match_clues. Explain the risk mechanism without metric evidence in the narrative.
match_clues is machine-parsed and comma-separated. Include relation/entity keywords plus optional structural constraints: first_hop=payer|payee|identity|fund|operation|object; has_identity=yes|no; has_fund=yes|no; has_operation=yes|no; has_object=yes|no; terminal=<entity>; value=count ratio. Distinct paradigms must have distinguishable structural constraints.
Split overly broad paradigms only into mutually exclusive structures, merge duplicates, and remove only disproven rules. Preserve unchanged rules by omission. Avoid concrete node IDs, absolute amounts, and one-off instances.
Return JSON only: {"rules":[{"slug":"","title":"","desc":"","structure":"","substructures":"","clues":"","risk":"","boundary":"","match_clues":""}],"remove":[],"changelog":[]} or {"unchanged":true,"reason":""}."""
