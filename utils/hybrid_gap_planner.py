"""
Diagnostic hybrid gap planner (v2).

Ranks minimal original-code snippets to add to a specification when regeneration
stalls below target similarity. Uses AST body diff, diff-line analysis, and optional
test-failure hints. Supports escalation from minimal lines to full statement blocks.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from utils.code_diff_analyzer import CodeDiffAnalyzer


_CATEGORY_WEIGHT = {
    "raise": 95.0,
    "except": 88.0,
    "control_flow": 82.0,
    "return": 78.0,
    "call": 72.0,
    "assign": 55.0,
    "docstring": 50.0,
    "literal_hint": 70.0,
    "statement_block": 65.0,
    "diff_line": 45.0,
    "smart_tier": 40.0,
}


@dataclass
class HybridGapCandidate:
    candidate_id: str
    code: str
    category: str
    score: float
    description: str
    source: str
    stmt_key: str = ""
    escalation_level: int = 0  # 0=minimal, 1=full statement block


@dataclass
class HybridGapPlan:
    candidates: List[HybridGapCandidate] = field(default_factory=list)
    gap_summary: List[str] = field(default_factory=list)


def _coerce_source(raw: str) -> str:
    from agents.advanced_analyzer import SemanticSimilarityAnalyzer

    return SemanticSimilarityAnalyzer.coerce_python_source_for_ast(raw or "")


def _unparse(node: ast.AST) -> str:
    if hasattr(ast, "unparse"):
        try:
            return ast.unparse(node).strip()
        except Exception:
            pass
    return ""


def _normalize_stmt(text: str) -> str:
    return " ".join((text or "").split())


def _candidate_id(code: str, stmt_key: str = "", level: int = 0) -> str:
    payload = f"{level}|{stmt_key}|{_normalize_stmt(code)}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _get_function_node(tree: ast.AST) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
    if isinstance(tree, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return tree
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None


def _categorize_stmt(stmt: ast.stmt) -> str:
    if isinstance(stmt, ast.Raise):
        return "raise"
    if isinstance(stmt, (ast.If, ast.While, ast.For)):
        return "control_flow"
    if isinstance(stmt, ast.Return):
        return "return"
    if isinstance(stmt, ast.Assign):
        return "assign"
    if isinstance(stmt, ast.Expr):
        val = stmt.value
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            return "docstring"
        if isinstance(val, ast.Call):
            return "call"
    if isinstance(stmt, ast.Try):
        return "except"
    return "statement_block"


def _minimal_line_from_stmt(stmt: ast.stmt, full_text: str) -> str:
    lines = full_text.splitlines()
    if not lines:
        return full_text.strip()
    first = lines[0].strip()
    if isinstance(stmt, (ast.If, ast.While, ast.For)):
        return first
    if isinstance(stmt, ast.Return):
        return first
    if isinstance(stmt, ast.Raise):
        return first
    if len(full_text) > 120 and isinstance(stmt, ast.Assign):
        lit = CodeDiffAnalyzer._extract_distinctive_literal_snippets(full_text, "")
        if lit:
            return lit
        return first
    return full_text.strip()


def _stmt_key(stmt: ast.stmt) -> str:
    return f"{type(stmt).__name__}:{getattr(stmt, 'lineno', 0)}"


def _regen_contains_approx(regen_norm: str, fragment: str) -> bool:
    frag = _normalize_stmt(fragment)
    if not frag or len(frag) < 4:
        return False
    if frag in regen_norm:
        return True
    # Substantial overlap — regenerated may paraphrase
    words = [w for w in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", frag) if len(w) > 2]
    if not words:
        return False
    hits = sum(1 for w in words if w in regen_norm)
    return hits / len(words) >= 0.85


def _test_failure_blob(test_data: Optional[Dict[str, Any]]) -> str:
    if not test_data:
        return ""
    parts: List[str] = []
    for f in test_data.get("failures") or []:
        if not isinstance(f, dict):
            continue
        t = f.get("test") or {}
        if isinstance(t, dict):
            parts.append(str(t.get("test_name", "")))
        parts.append(str(f.get("reason", "")))
        parts.append(str(f.get("regenerated_output", "")))
    for nid in test_data.get("pytest_regen_failed_nodeids") or []:
        parts.append(str(nid))
    return " ".join(parts).lower()


def _test_relevance_boost(code: str, test_blob: str) -> float:
    if not test_blob:
        return 0.0
    boost = 0.0
    for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", code.lower()):
        if token in test_blob:
            boost += 4.0
    return min(boost, 24.0)


def _size_penalty(code: str, original_len: int) -> float:
    if original_len <= 0:
        return 0.0
    ratio = len(code) / original_len
    return min(35.0, ratio * 40.0)


def _score_candidate(
    code: str,
    category: str,
    original_len: int,
    regen_norm: str,
    test_blob: str,
    escalation_level: int,
) -> float:
    base = _CATEGORY_WEIGHT.get(category, 40.0)
    if escalation_level > 0:
        base += 8.0
    score = base - _size_penalty(code, original_len) + _test_relevance_boost(code, test_blob)
    if _regen_contains_approx(regen_norm, code):
        score -= 25.0
    return score


def _ast_missing_candidates(
    original_code: str,
    regenerated_code: str,
    *,
    escalate_stmt_keys: Set[str],
    test_blob: str,
) -> List[HybridGapCandidate]:
    out: List[HybridGapCandidate] = []
    orig_src = _coerce_source(original_code)
    regen_src = _coerce_source(regenerated_code)
    if not orig_src or not regen_src:
        return out
    try:
        orig_tree = ast.parse(orig_src)
        regen_tree = ast.parse(regen_src)
    except SyntaxError:
        return out

    orig_fn = _get_function_node(orig_tree)
    regen_fn = _get_function_node(regen_tree)
    if not orig_fn or not regen_fn:
        return out

    regen_norm = _normalize_stmt(_unparse(regen_fn))
    orig_len = len(orig_src)

    for stmt in orig_fn.body:
        full_text = _unparse(stmt)
        if not full_text:
            continue
        norm = _normalize_stmt(full_text)
        if norm in {_normalize_stmt(_unparse(s)) for s in regen_fn.body}:
            continue
        if _regen_contains_approx(regen_norm, full_text):
            continue

        sk = _stmt_key(stmt)
        cat = _categorize_stmt(stmt)
        level = 1 if sk in escalate_stmt_keys else 0
        code = full_text if level > 0 else _minimal_line_from_stmt(stmt, full_text)
        desc = f"Missing {cat}: {code[:60]}..."
        cid = _candidate_id(code, sk, level)
        score = _score_candidate(code, cat, orig_len, regen_norm, test_blob, level)
        out.append(
            HybridGapCandidate(
                candidate_id=cid,
                code=code,
                category=cat,
                score=score,
                description=desc,
                source="ast_body_diff",
                stmt_key=sk,
                escalation_level=level,
            )
        )
    return out


def _diff_line_candidates(
    original_code: str,
    regenerated_code: str,
    already_norm: Set[str],
    test_blob: str,
) -> List[HybridGapCandidate]:
    out: List[HybridGapCandidate] = []
    orig_len = len(original_code or "")
    regen_norm = _normalize_stmt(_coerce_source(regenerated_code))
    for piece in CodeDiffAnalyzer.get_diff_driven_pieces(
        original_code, regenerated_code, list(already_norm), max_pieces=12
    ):
        code = str(piece.get("code", "")).strip()
        if not code:
            continue
        norm = _normalize_stmt(code)
        if norm in already_norm:
            continue
        if CodeDiffAnalyzer._is_docstring_piece(code):
            cat = "docstring"
        elif code.startswith("#"):
            cat = "literal_hint"
        else:
            cat = "diff_line"
        score = float(piece.get("priority", 40)) + _test_relevance_boost(code, test_blob)
        score -= _size_penalty(code, orig_len)
        if _regen_contains_approx(regen_norm, code):
            continue
        out.append(
            HybridGapCandidate(
                candidate_id=_candidate_id(code),
                code=code,
                category=cat,
                score=score,
                description=str(piece.get("description", "Diff line")),
                source="diff_line",
            )
        )
    return out


def _smart_fallback_candidates(
    original_code: str,
    regenerated_code: str,
    already_added: List[str],
    test_blob: str,
) -> List[HybridGapCandidate]:
    from utils.smart_code_extractor import SmartCodeExtractor

    out: List[HybridGapCandidate] = []
    orig_len = len(original_code or "")
    regen_norm = _normalize_stmt(_coerce_source(regenerated_code))
    for piece in SmartCodeExtractor().get_next_pieces(
        original_code, already_added, regenerated_code, max_pieces_per_iter=8
    ):
        code = piece.code.strip()
        if not code or _regen_contains_approx(regen_norm, code):
            continue
        score = float(piece.priority) + _test_relevance_boost(code, test_blob)
        score -= _size_penalty(code, orig_len)
        out.append(
            HybridGapCandidate(
                candidate_id=_candidate_id(code, piece.tier),
                code=code,
                category="smart_tier",
                score=score,
                description=piece.description or piece.tier,
                source="smart_extractor",
                stmt_key=piece.tier,
            )
        )
    return out


class HybridGapPlanner:
    """Plan ranked hybrid additions for one function."""

    def plan(
        self,
        original_code: str,
        regenerated_code: str,
        *,
        already_added: Optional[List[str]] = None,
        rejected_ids: Optional[Set[str]] = None,
        escalate_stmt_keys: Optional[Set[str]] = None,
        test_data: Optional[Dict[str, Any]] = None,
        max_candidates: int = 20,
    ) -> HybridGapPlan:
        already = already_added or []
        already_norm = {_normalize_stmt(a) for a in already}
        rejected = rejected_ids or set()
        escalate = escalate_stmt_keys or set()
        test_blob = _test_failure_blob(test_data)

        merged: Dict[str, HybridGapCandidate] = {}
        for cand in (
            _ast_missing_candidates(
                original_code,
                regenerated_code,
                escalate_stmt_keys=escalate,
                test_blob=test_blob,
            )
            + _diff_line_candidates(original_code, regenerated_code, already_norm, test_blob)
            + _smart_fallback_candidates(original_code, regenerated_code, already, test_blob)
        ):
            if cand.candidate_id in rejected:
                continue
            if _normalize_stmt(cand.code) in already_norm:
                continue
            prev = merged.get(cand.candidate_id)
            if prev is None or cand.score > prev.score:
                merged[cand.candidate_id] = cand

        has_behavioral_ast = any(
            c.source == "ast_body_diff"
            and c.category in ("raise", "control_flow", "except", "return", "call", "statement_block")
            for c in merged.values()
        )
        if has_behavioral_ast:
            merged = {
                k: v
                for k, v in merged.items()
                if v.category not in ("docstring", "smart_tier")
                and not CodeDiffAnalyzer._is_docstring_piece(v.code)
            }

        def _sort_key(c: HybridGapCandidate) -> Tuple[float, int]:
            src_rank = {"ast_body_diff": 0, "diff_line": 1, "smart_extractor": 2}.get(c.source, 3)
            return (-c.score, src_rank)

        ranked = sorted(merged.values(), key=_sort_key)[:max_candidates]
        summaries = CodeDiffAnalyzer.get_diff_natural_language_descriptions(
            original_code, regenerated_code, max_descriptions=8
        )
        return HybridGapPlan(candidates=ranked, gap_summary=summaries)

    @staticmethod
    def append_addition(spec: Dict[str, Any], code: str) -> None:
        """Append one hybrid snippet to spec with docstring dedup."""
        if "hybrid_code_additions" not in spec:
            spec["hybrid_code_additions"] = []
        existing = spec["hybrid_code_additions"]
        code = code.strip()
        if not code:
            return
        if CodeDiffAnalyzer._is_docstring_piece(code):
            content = CodeDiffAnalyzer._docstring_content(code)
            for ex in existing:
                if CodeDiffAnalyzer._is_docstring_piece(str(ex)):
                    if CodeDiffAnalyzer._docstring_content(str(ex)) == content:
                        return
        if code not in [str(ex).strip() for ex in existing]:
            existing.append(code)

    @staticmethod
    def remove_last_addition(spec: Dict[str, Any], code: str) -> None:
        additions = spec.get("hybrid_code_additions") or []
        code = code.strip()
        if code in [str(a).strip() for a in additions]:
            spec["hybrid_code_additions"] = [a for a in additions if str(a).strip() != code]
