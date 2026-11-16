from __future__ import annotations

import argparse
import json
import math
import random
import re
from datetime import datetime
from pathlib import Path

from system.evolution.evaluator_eval import STATIC_AGENT_SAMPLE_SIZE, validate_static_assets
from system.paths import PROJECT_ROOT


WORKSPACE = PROJECT_ROOT
BOOK_FILES = {"graph_paradigm": "graph_paradigm.md",
              "explore_navigate": "explore_navigate.md",
              "feature_hypothesize": "feature_hypothesize.md"}
SAMPLE_SEED = 20260817
CARD_COUNTS = {"paradigm": 6, "navigate": 8, "hypothesize": 10,
               "synthesis": 8, "eval": 8}
METHOD_PREFIXES = ("manual", "stat", "comm")
TIER_LABEL = {"high": "high", "mid": "mid", "low": "low"}


def _iv_tier(value) -> str:
    from system.core.schema import IVTier
    return IVTier.from_value(float(value)).value


def _read_pool(source: Path) -> list[dict]:
    path = source / "features.json" if source.is_dir() else source
    if not path.is_file():
        raise SystemExit(f"Baseline pool not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("features") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise SystemExit(f"Invalid baseline pool: {path}")
    return [dict(row, _source_run=source.parent.name if source.is_file() else source.name)
            for row in rows if isinstance(row, dict)]


def _memory_item(row: dict) -> dict | None:
    name = str(row.get("feature_name") or "").strip()
    try:
        true_iv = float(row.get("true_iv"))
    except (TypeError, ValueError):
        return None
    logic = str(row.get("logic") or "").strip()
    if not name or not logic or not math.isfinite(true_iv) or true_iv < 0:
        return None
    risk = str(row.get("business_explanation") or row.get("feature_understanding")
               or row.get("fraud_pattern") or row.get("risk_pattern") or "").strip()
    desc = f"Feature: {name}\nStructure: {logic}\nRisk rationale: {risk}"
    if re.search(r"[\u4e00-\u9fff]", "\n".join((name, logic, risk))):
        raise SystemExit(f"Static-Agent source is not English: {name}")
    return {"doc_name": name, "feature_desc": desc, "true_iv": true_iv,
            "tier": _iv_tier(true_iv), "logic": logic,
            "evidence_chain": row.get("evidence_chain") or [],
            "expected_value_type": row.get("expected_value_type") or "",
            "source_method": str(row.get("_source_method") or ""),
            "source_run": str(row.get("_source_run") or "")}


def sample_eval_memory(sources: list[Path], n: int = STATIC_AGENT_SAMPLE_SIZE,
                       seed: int = SAMPLE_SEED) -> list[dict]:
    if n != STATIC_AGENT_SAMPLE_SIZE:
        raise SystemExit(f"The paper Static-Agent protocol requires exactly {STATIC_AGENT_SAMPLE_SIZE} sampled features")
    by_name: dict[str, list[dict]] = {}
    found_methods: set[str] = set()
    for source in sources:
        run_name = source.parent.name if source.is_file() else source.name
        method = next((name for name in METHOD_PREFIXES
                       if run_name == name or run_name.startswith(name + "__")), "")
        if not method:
            raise SystemExit(f"Not a Manual/Stat/Comm baseline run: {source}")
        found_methods.add(method)
        for row in _read_pool(source):
            row["_source_method"] = method
            item = _memory_item(row)
            if item is not None:
                by_name.setdefault(item["doc_name"], []).append(item)
    if found_methods != set(METHOD_PREFIXES):
        raise SystemExit("Static-Agent requires real feature pools from Manual, Stat, and Comm")
    if len(by_name) < n:
        raise SystemExit(f"Need at least {n} distinct measured baseline features; "
                         f"found {len(by_name)} across Manual, Stat, and Comm")
    rng = random.Random(seed)
    names = rng.sample(sorted(by_name), n)
    picked = [rng.choice(by_name[name]) for name in names]
    if {item["source_method"] for item in picked} != found_methods:
        raise SystemExit("The sampled 60 baseline features must include all three source methods")
    print(f"Uniformly sampled {len(picked)} of {len(by_name)} distinct measured baseline features "
          f"with seed={seed}")
    return picked


def _field_of(desc: str, key: str) -> str:
    for line in str(desc or "").splitlines():
        if line.strip().lower().startswith(key.lower() + ":"):
            return line.split(":", 1)[1].strip()
    return ""


def _sample_material(picked: list[dict]) -> str:
    lines = ["[Static-Agent SOURCE MATERIAL]",
             f"These {len(picked)} uniformly sampled, legally pooled Prompt-baseline features are "
             "the sole experience source for every frozen playbook and the IV-RAG memory. "
             "Abstract structure-level lessons; do not copy feature names, anonymous node IDs, "
             "absolute amounts, sample indices, or batch ranges."]
    order = list(range(len(picked)))
    random.Random(SAMPLE_SEED).shuffle(order)
    for idx in order:
        item = picked[idx]
        structure = _field_of(item.get("feature_desc", ""), "Structure") or item.get("logic") or ""
        risk = _field_of(item.get("feature_desc", ""), "Risk rationale")
        lines.append(f"- structure: {structure} | external_tier: {TIER_LABEL[item['tier']]}"
                     + (f" | risk: {risk}" if risk else ""))
    return "\n".join(lines)


MATERIAL_TAIL = {
    "paradigm": "Every paradigm must correspond to an observed structural family in the material. "
                "Use exact schema fragments in match_clues and mutually exclusive structured bases.",
    "explore": "Navigation rules must explain how to complete an observed structural family. "
               "Hypothesis rules must define the statistic, scalar form, discrimination mechanism, and boundary.",
    "synthesis": "Infer reusable Gremlin plan forms and failure avoidance from the relation combinations. "
                 "Do not write business-risk judgments.",
    "eval": "Use external_tier as calibration evidence. Explain the likely estimator bias by structural family "
            "without reporting batch statistics or exact measured values.",
}


def _book_material(material: str, kind: str) -> str:
    return material + "\n" + MATERIAL_TAIL[kind]


REDLINE = """[Static-Agent MATERIAL BOUNDARY]
Use only the supplied frozen source material and graph schema. Do not invent unrelated structural families. Do not copy concrete feature names, anonymous node IDs, absolute amounts, run IDs, episode numbers, or exact measured IV values. Use transferable class-level statements and exact English schema identifiers where required."""


CARD_USER = """Initialize an empty playbook with at most __N__ entries.
__REDLINE__

[GRAPH SCHEMA]
__VOCAB__
__EXTRA__
Return the JSON schema required by the system prompt, with top-level {"rules": [...]} and snake_case slugs. Prefer fewer evidence-backed entries over padding or duplication."""
SYNTH_USER = """Initialize an empty synthesis playbook with at most __N__ entries covering distinct anchor families.
__REDLINE__
Focus on executable Gremlin plans, stored direction, reductions, and common implementation failures rather than business-risk semantics.

[GRAPH SCHEMA]
__VOCAB__
__EXTRA__
Return {"skills":[{"anchor","title","when","do","avoid","example","mechanism","boundary"}]} with anchors from the system-provided families."""


EVAL_USER = """Initialize an empty evaluator-calibration playbook with at most __N__ entries for distinct structural families.
__REDLINE__
Explain the external tier, likely prediction bias, calibration action, and applicability boundary without exact values or instance-specific names.

[GRAPH SCHEMA]
__VOCAB__
__EXTRA__
Return the system-required JSON with top-level {"rules": [...]} and snake_case slugs."""


PARADIGM_KW_RULE = """[MATCH_CLUES DISCIPLINE]
Every keyword must be an exact substring of the supplied schema, such as a public relation name fragment, payer/payee, or entity ID/short name. Use at least two live keywords per paradigm. Structured bases use first_hop=..., terminal=..., has_identity=true|false, has_fund=true|false, has_operation=true|false, and has_object=true|false. Never invent relation abbreviations."""


def _fill(template: str, n: int, vocab: str, extra: str = "") -> str:
    return (template.replace("__N__", str(n)).replace("__REDLINE__", REDLINE)
            .replace("__VOCAB__", vocab).replace("__EXTRA__", extra))


def _schema_vocab() -> str:
    from system.core.vocabulary import (entity_category, entity_definitions, public_entity_fields,
                                        public_relation_bindings, public_relation_endpoints)
    lines = ["Entity types use the exact snake_case identifiers listed below.",
             "relation uses the exact public relation name; payer_side and payee_side mark source roles.",
             "[ENTITIES: entity_type | category | fields]"]
    for entity_id in sorted(entity_definitions()):
        lines.append(f"{entity_id} | {entity_category(entity_id)} | "
                     f"{','.join(sorted(public_entity_fields(entity_id)))}")
    lines.append("[RELATIONS: relation | source->target | stored direction]")
    for fields in public_relation_bindings().values():
        for relation in sorted(fields):
            source, target = public_relation_endpoints(relation)
            lines.append(f"{relation} | {source}->{target} | out")
    lines.append("Value/reduction types: count, flag, ratio, amount, scalar, reduce, exists")
    return "\n".join(lines)


def _vocab_blob() -> str:
    from system.core.vocabulary import (entity_definitions, public_entity_fields,
                                        public_relation_bindings)
    parts = []
    for fields in public_relation_bindings().values():
        parts.extend(str(relation) for relation in fields)
    for entity_id in entity_definitions():
        parts.extend([entity_id, *public_entity_fields(entity_id)])
    parts.extend(["count", "flag", "ratio", "amount", "scalar", "reduce", "exists"])
    return " ".join(x for x in parts if x).lower()


CLUE_STRUCT_PREFIXES = ("first_hop=", "terminal=", "value=", "has_identity=",
                        "has_fund=", "has_operation=", "has_object=")


def _live_keywords(text: str, blob: str) -> tuple[list[str], list[str]]:
    normalized = re.sub(r"[,;/|]", ",", str(text or ""))
    live, dead = [], []
    for item in normalized.split(","):
        item = item.strip()
        if not item:
            continue
        if any(item.lower().startswith(prefix) for prefix in CLUE_STRUCT_PREFIXES):
            continue
        (live if item.lower() in blob else dead).append(item)
    return live, dead


def _repair_dead_clues(llm, rules: list[dict], blob: str) -> None:
    bad = []
    for rule in rules:
        live, dead = _live_keywords(rule.get("match_clues") or "", blob)
        if not live:
            bad.append({"slug": rule.get("slug"), "title": rule.get("title"),
                        "match_clues": rule.get("match_clues"), "dead_keywords": dead})
    if not bad:
        return
    instruction = ("Rewrite only match_clues so every paradigm has at least one exact substring from the supplied "
                   "schema corpus. Preserve structured bases and slugs. Return "
                   '{"rules":[{"slug":"...","match_clues":"..."}]}' + "\n\nSchema corpus:\n"
                   + blob[:6000] + "\n\nRules to repair:\n"
                   + json.dumps(bad, ensure_ascii=False, indent=1))
    response = llm.complete_json("Repair deterministic graph-paradigm matching clues.", instruction,
                                 temperature=0.2)
    fixed = {str(x.get("slug") or ""): str(x.get("match_clues") or "")
             for x in ((response or {}).get("rules") or []) if isinstance(x, dict)}
    for rule in rules:
        replacement = fixed.get(str(rule.get("slug") or ""))
        if replacement:
            rule["match_clues"] = replacement


def _paradigm_block(memory) -> str:
    from system.evolution.skill_memory import PARADIGM_ID
    lines = ["[CURRENT PARADIGMS: reference only these slugs]"]
    for _, card in memory._books[PARADIGM_ID].iter_cards():
        lines.append(f"- {card.title or card.slug} ({card.slug})")
        for field in ("structure", "substructures"):
            value = str(card.field(field) or "").strip()
            if value:
                lines.append(f"  {field}: {value}")
    return "\n".join(lines)


def rewrite_books(out: Path, llm, counts: dict, material: str) -> None:
    from system.evolution.eval_skill_memory import EvalSkillMemory
    from system.evolution.eval_skill_memory import _REVISE_SYSTEM as EVAL_SYSTEM
    from system.evolution.skill_memory import (HYPOTHESIZE_ID, NAVIGATE_ID, PARADIGM_ID,
                                               SkillMemory, _REVISE_SYSTEM)
    from system.evolution.synthesis_skill_memory import SynthesisSkillMemory, _SYSTEM as SYN_SYSTEM
    vocab = _schema_vocab()
    memory = SkillMemory(llm, store_path=out, persist=True, reset=True)

    def ask(system_prompt, user_prompt, kind):
        response = llm.complete_json(system_prompt, user_prompt, temperature=0.6)
        rules = (response or {}).get("rules") if isinstance(response, dict) else None
        if not rules:
            raise SystemExit(f"Static-Agent {kind} generation returned no valid rules")
        return rules

    rules = ask(_REVISE_SYSTEM[PARADIGM_ID],
                _fill(CARD_USER, counts["paradigm"], vocab,
                      PARADIGM_KW_RULE + "\n" + _book_material(material, "paradigm")), "paradigm")
    blob = _vocab_blob()
    _repair_dead_clues(llm, rules, blob)
    memory.seed_book(PARADIGM_ID, rules, source="Static-Agent initialization",
                     note="Same 60 uniformly sampled Prompt-baseline features as IV-RAG")
    memory.register_paradigms()
    paradigm_block = _paradigm_block(memory)
    for book_id, key in ((NAVIGATE_ID, "navigate"), (HYPOTHESIZE_ID, "hypothesize")):
        rules = ask(_REVISE_SYSTEM[book_id],
                    _fill(CARD_USER, counts[key], vocab,
                          paradigm_block + "\n" + _book_material(material, "explore")), key)
        memory.seed_book(book_id, rules, source="Static-Agent initialization",
                         note="Same frozen source sample as IV-RAG")
    memory.save()
    synthesis = SynthesisSkillMemory(llm, store_path=out / "synthesis_playbook.md",
                                     persist=True, reset=True)
    response = llm.complete_json(SYN_SYSTEM, _fill(SYNTH_USER, counts["synthesis"], vocab,
                                                  _book_material(material, "synthesis")),
                                 temperature=0.6)
    skills = (response or {}).get("skills") if isinstance(response, dict) else None
    if not skills:
        raise SystemExit("Static-Agent synthesis generation returned no valid skills")
    synthesis.seed_skills(skills, source="Static-Agent initialization",
                          note="Same frozen source sample as IV-RAG")
    evaluation = EvalSkillMemory(llm, store_path=out / "eval_calibration.md",
                                 persist=True, reset=True)
    rules = ask(EVAL_SYSTEM, _fill(EVAL_USER, counts["eval"], vocab,
                                   _book_material(material, "eval")), "eval")
    evaluation.seed_rules(rules, source="Static-Agent initialization",
                          note="Same frozen source sample as IV-RAG")


def _default_sources() -> list[Path]:
    runs = WORKSPACE / "runs"
    sources = []
    for method in METHOD_PREFIXES:
        sources.extend(sorted(path for path in runs.glob(method + "__s*") if path.is_dir()))
    if not sources:
        raise SystemExit("Provide --baseline-run for Manual, Stat, and Comm pools")
    return sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", action="append", default=[],
                        help="Baseline run directory or features.json; repeat for every source pool")
    parser.add_argument("--out", default=str(WORKSPACE / "config" / "static_agent"))
    parser.add_argument("--sample-size", type=int, default=STATIC_AGENT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    for key in CARD_COUNTS:
        parser.add_argument("--cards-" + key, type=int, default=CARD_COUNTS[key])
    args = parser.parse_args()
    if args.sample_size != STATIC_AGENT_SAMPLE_SIZE:
        raise SystemExit(f"The paper Static-Agent protocol requires exactly {STATIC_AGENT_SAMPLE_SIZE} sampled features")
    sources = [Path(x) for x in args.baseline_run] or _default_sources()
    picked = sample_eval_memory(sources, args.sample_size, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_memory.json").write_text(
        json.dumps({"updated_at": datetime.now().isoformat(timespec="seconds"),
                    "count": len(picked), "sample_seed": args.seed,
                    "source_methods": list(METHOD_PREFIXES), "examples": picked},
                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    from system.llm.client import LLMClient
    counts = {key: getattr(args, "cards_" + key) for key in CARD_COUNTS}
    rewrite_books(out, LLMClient(name="static_agent_materialize"), counts, _sample_material(picked))
    validate_static_assets(out)
    print(f"Static-Agent assets written to {out}")


if __name__ == "__main__":
    main()
