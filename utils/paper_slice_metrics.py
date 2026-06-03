"""
Static metrics for defining an in-scope “paper slice” (size, AST depth, dynamism).

Used by ``scripts/paper_slice_report.py``. No LLM dependency.
"""

from __future__ import annotations

import ast
import re
import textwrap
from typing import Any, Dict, List, Tuple

_RISKY_CALL_NAMES = frozenset(
    {
        "exec",
        "eval",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "locals",
        "globals",
    }
)


def _non_empty_lines(source: str) -> int:
    return sum(1 for ln in source.splitlines() if ln.strip())


def ast_max_depth(root: ast.AST) -> int:
    deepest = 0
    stack: List[Tuple[ast.AST, int]] = [(root, 1)]
    while stack:
        node, d = stack.pop()
        deepest = max(deepest, d)
        for child in ast.iter_child_nodes(node):
            stack.append((child, d + 1))
    return deepest


def dynamism_signal_count_ast(tree: ast.AST) -> int:
    """Heuristic counts of obvious dynamic/metaprogramming-style calls."""
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _RISKY_CALL_NAMES:
                n += 1
    return n


_FALLBACK_DYNAMIC_PATTERNS = (
    (re.compile(r"\bexec\s*\("), "exec"),
    (re.compile(r"\beval\s*\("), "eval"),
    (re.compile(r"\b__import__\s*\("), "__import__"),
    (re.compile(r"\bglobals\s*\(\s*\)"), "globals"),
    (re.compile(r"\blocals\s*\(\s*\)"), "locals"),
    (re.compile(r"\bgetattr\s*\("), "getattr"),
    (re.compile(r"\bsetattr\s*\("), "setattr"),
    (re.compile(r"\bhasattr\s*\("), "hasattr"),
)


def dynamism_fallback_regex(source: str) -> int:
    """Count rough matches when AST parse fails (non-compositional)."""
    c = 0
    for rx, _ in _FALLBACK_DYNAMIC_PATTERNS:
        c += len(rx.findall(source))
    return c


def compute_paper_metrics(original_code: str) -> Dict[str, Any]:
    """
    Metrics for slicing. ``parse_ok`` false means structural metrics are unreliable;
    fallback regex contributes ``dynamic_signals_regex`` only.
    """
    stripped = original_code.strip() if original_code else ""
    out: Dict[str, Any] = {
        "source_chars": len(stripped),
        "source_lines_non_empty": _non_empty_lines(stripped),
        "parse_ok": False,
        "ast_max_depth": 0,
        "dynamic_signals_ast": 0,
        "dynamic_signals_regex": dynamism_fallback_regex(stripped),
        "dynamic_signals": 0,
    }
    if not stripped:
        return out

    try:
        tree = ast.parse(textwrap.dedent(stripped))
    except SyntaxError:
        out["dynamic_signals"] = out["dynamic_signals_regex"]
        return out

    out["parse_ok"] = True
    out["ast_max_depth"] = ast_max_depth(tree)
    ast_dyn = dynamism_signal_count_ast(tree)
    out["dynamic_signals_ast"] = ast_dyn
    out["dynamic_signals"] = ast_dyn + out["dynamic_signals_regex"]
    return out
