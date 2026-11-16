from __future__ import annotations


_CLASS_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("relation", ("invented", "absent from the evidence chain", "relation '")),
    ("depth", ("hops", "edge-expansion", "step_plan")),
    ("aggregate", ("expected_value_type", "incompatible", "aggregate", "final")),
    ("exec", ("execution failed", "timed out", "timeout", "slow query", "server unavailable")),
    ("consistency", ("consistency", "inconsistent", "element_comparison", "measured object")),
    ("query_form", ("no plan", "syntax", "placeholder", "anchor", "scan the whole graph",
                    "depends_on", "cycle")),
]


def error_signature(text: str) -> str:
    classes = [name for name, kws in _CLASS_KEYWORDS
               if any(k in text for k in kws)]
    return "+".join(classes) if classes else "unknown"
