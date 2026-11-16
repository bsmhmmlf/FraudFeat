from __future__ import annotations

import json
import sys
from pathlib import Path

from ._common import emit, write_txt


def _load_cases(run_dir: Path) -> list[dict]:
    case_dir = run_dir / "cases"
    if not case_dir.is_dir():
        return []
    out = []
    for p in sorted(case_dir.glob("w*_*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return out


def build_index(run_dir: str | Path) -> list[str]:
    run_dir = Path(run_dir)
    cases = _load_cases(run_dir)
    lines: list[str] = []
    lines.append(emit("cases_total", value=len(cases)))
    full = [c for c in cases
            if c.get("outcome") == "pooled" and c.get("trace")]
    lines.append(emit("case_type1_full_trajectory", value=len(full),
                      names=",".join(c["feature_name"] for c in full[:8]) or None))
    fail = [c for c in cases
            if c.get("outcome") in ("synth_failed", "frozen")]
    lines.append(emit("case_type2_failure", value=len(fail),
                      names=",".join(c["feature_name"] for c in fail[:8]) or None))
    by_outcome: dict[str, str] = {}
    for c in cases:
        key = c.get("outcome")
        if key in ("synth_failed", "frozen") and key not in by_outcome:
            by_outcome[key] = c["feature_name"]
    for k, v in sorted(by_outcome.items()):
        lines.append(emit("case_type5_error_example", outcome=k, feature=v))
    cards = [c for c in cases
             if (c.get("entry") or {}).get("business_explanation")]
    lines.append(emit("case_type3_explain_card", value=len(cards)))
    prog = run_dir / "progress.json"
    n_skills = 0
    if prog.is_file():
        n_skills = len(json.loads(prog.read_text(encoding="utf-8"))
                       .get("skills") or [])
    lines.append(emit("case_type4_skill_entries", value=n_skills,
                      note="progress.json.skills"))
    return lines


def main(argv: list[str]) -> None:
    for d in argv:
        lines = build_index(d)
        out = write_txt(d, "cases_index.txt", lines)
        print(f"[cases] {d} → {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main(sys.argv[1:])
