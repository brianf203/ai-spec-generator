"""
Classify diff lines for hybrid / round-trip failure analysis (paper: taxonomy of gaps).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Any


def classify_line(line: str) -> str:
    """Assign a coarse category to a single source line (heuristic)."""
    s = line.strip()
    if not s:
        return "blank"
    if s.startswith("#"):
        return "comment"
    if s.startswith(("import ", "from ")):
        return "import"
    low = s.lower()
    if low.startswith(("def ", "async def ", "class ")):
        return "signature_or_definition"
    if re.search(r'raise\s+\w', low):
        return "raise_or_exception"
    if "try:" in low or "except" in low or "finally:" in low:
        return "try_except"
    if re.match(r"^(if|elif|else|for|while|with)\b", low):
        return "control_structure"
    if re.search(r"['\"].{12,}['\"]", s) or s.count('"') + s.count("'") >= 4:
        return "string_or_literal_heavy"
    if re.search(r"\b\d+\.\d+|\b0x[0-9a-fA-F]+", s) and "=" in s:
        return "numeric_literal"
    if re.search(r"\w+\s*\([^)]*\)", s) and "=" in s:
        return "call_or_assignment"
    if "=" in s and not s.startswith("return"):
        return "assignment"
    if s.startswith("return"):
        return "return"
    return "other"


def aggregate_taxonomy(lines: List[str]) -> Dict[str, Any]:
    """Count categories and return proportions."""
    counts = Counter(classify_line(ln) for ln in lines)
    total = sum(counts.values()) or 1
    return {
        "counts": dict(counts),
        "fractions": {k: v / total for k, v in counts.items()},
        "total_lines": total,
    }


def taxonomy_from_missing_lines(missing_lines: List[str]) -> Dict[str, Any]:
    """Build taxonomy from CodeDiffAnalyzer missing_lines (original-only lines)."""
    return aggregate_taxonomy(missing_lines)
