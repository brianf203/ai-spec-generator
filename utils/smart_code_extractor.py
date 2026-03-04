"""
Smart code extraction for hybrid specs - prioritizes pieces by constraint strength.
Research-grade: adds minimal code in optimal order to maximize similarity gain per addition.
"""
import ast
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CodePiece:
    """A minimal code piece to add, with priority for ordering."""
    code: str
    priority: int  # Higher = add first (more constraining)
    tier: str  # For logging: 'anchor', 'control_flow', 'expression', etc.
    description: str


class SmartCodeExtractor:
    """
    Extracts code pieces from original in order of constraint strength.
    Strategy: Add the most constraining pieces first - string literals, docstrings,
    control flow, return expressions - so each addition maximizes similarity gain.
    """

    @staticmethod
    def extract_prioritized_pieces(original_code: str) -> List[CodePiece]:
        """
        Extract code pieces from original, ordered by constraint strength.
        Returns list of (code, priority, tier) - add in order, one or few per iteration.
        """
        pieces: List[CodePiece] = []
        if not original_code or not original_code.strip():
            return pieces

        try:
            tree = ast.parse(original_code)
        except SyntaxError:
            return SmartCodeExtractor._fallback_line_based(original_code)

        func = SmartCodeExtractor._get_function(tree)
        if not func:
            return SmartCodeExtractor._fallback_line_based(original_code)

        lines = original_code.split('\n')

        # Tier 1: Docstring - high impact, usually small
        for stmt in func.body:
            val = None
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                val = stmt.value.value
            if isinstance(val, str):
                if hasattr(stmt, 'lineno') and stmt.lineno and lines:
                    lo, hi = stmt.lineno - 1, (stmt.end_lineno or stmt.lineno) - 1
                    doc_src = '\n'.join(lines[lo:hi + 1]).strip()
                    if doc_src:
                        pieces.append(CodePiece(
                            code=doc_src,
                            priority=100,
                            tier='docstring',
                            description='Exact docstring'
                        ))
                break

        # Tier 2: String literals in Raise - "Cannot divide by zero" etc.
        for node in ast.walk(func):
            if isinstance(node, ast.Raise):
                piece = SmartCodeExtractor._extract_raise_with_message(node, lines)
                if piece:
                    pieces.append(piece)

        # Tier 3: Return statements - exact expressions
        seen_returns = set()
        for node in ast.walk(func):
            if isinstance(node, ast.Return) and node.value is not None:
                piece = SmartCodeExtractor._extract_node_source(node, lines, 'return', 80)
                if piece and piece.strip() not in seen_returns:
                    seen_returns.add(piece.strip())
                    pieces.append(CodePiece(
                        code=piece,
                        priority=70,
                        tier='return',
                        description='Return expression'
                    ))

        # Tier 4: If/elif conditions - control flow structure
        seen_conditions = set()
        for node in ast.walk(func):
            if isinstance(node, ast.If):
                piece = SmartCodeExtractor._extract_if_condition(node, lines)
                if piece and piece.strip() not in seen_conditions:
                    seen_conditions.add(piece.strip())
                    pieces.append(CodePiece(
                        code=piece,
                        priority=60,
                        tier='control_flow',
                        description='Condition'
                    ))

        # Tier 5: Remaining statements - by line importance
        added_lines = set()
        for p in pieces:
            for line in p.code.split('\n'):
                added_lines.add(line.strip())

        body_lines = SmartCodeExtractor._get_body_lines(func, lines)
        for i, (line, score) in enumerate(body_lines):
            stripped = line.strip()
            if stripped and stripped not in added_lines:
                pieces.append(CodePiece(
                    code=stripped,
                    priority=50 - (i // 2),  # Earlier lines first
                    tier='statement',
                    description='Body statement'
                ))
                added_lines.add(stripped)

        # Sort by priority descending, then deduplicate by code
        seen = set()
        unique = []
        for p in sorted(pieces, key=lambda x: -x.priority):
            norm = p.code.strip()
            if norm and norm not in seen:
                seen.add(norm)
                unique.append(p)

        return unique

    @staticmethod
    def _get_function(tree: ast.AST) -> Optional[ast.FunctionDef]:
        """Extract the first function from module or class."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node
        return None

    @staticmethod
    def _extract_node_source(node: ast.AST, lines: List[str], prefix: str, default_priority: int) -> Optional[str]:
        """Extract source for a node using line numbers."""
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            lo, hi = node.lineno - 1, (node.end_lineno or node.lineno) - 1
            snippet = '\n'.join(lines[lo:hi + 1]).strip()
            return snippet if snippet else None
        if hasattr(ast, 'unparse'):
            try:
                return ast.unparse(node)
            except Exception:
                pass
        return None

    @staticmethod
    def _extract_raise_with_message(node: ast.Raise, lines: List[str]) -> Optional[CodePiece]:
        """Extract full raise statement with exact string message."""
        snippet = SmartCodeExtractor._extract_node_source(node, lines, 'raise', 90)
        if snippet:
            return CodePiece(
                code=snippet,
                priority=90,
                tier='raise',
                description='Raise with exact message'
            )
        return None

    @staticmethod
    def _extract_if_condition(node: ast.If, lines: List[str]) -> Optional[str]:
        """Extract if condition line only (minimal constraint)."""
        if hasattr(node, 'lineno') and node.lineno and lines:
            idx = node.lineno - 1
            if 0 <= idx < len(lines):
                line = lines[idx].strip()
                if line.startswith('if ') or line.startswith('elif '):
                    return line
        if hasattr(ast, 'unparse'):
            try:
                return f"if {ast.unparse(node.test)}:"
            except Exception:
                pass
        return None

    @staticmethod
    def _get_body_lines(func: ast.FunctionDef, lines: List[str]) -> List[Tuple[str, int]]:
        """Get (line, importance_score) for function body lines."""
        result = []
        for stmt in func.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue  # Skip docstring
            if hasattr(stmt, 'lineno') and hasattr(stmt, 'end_lineno'):
                lo, hi = stmt.lineno - 1, (stmt.end_lineno or stmt.lineno) - 1
                for i in range(lo, min(hi + 1, len(lines))):
                    line = lines[i]
                    score = 0
                    if 'return ' in line or 'raise ' in line:
                        score += 2
                    if any(c in line for c in ['"', "'"]):
                        score += 2
                    if any(c in line for c in '0123456789'):
                        score += 1
                    result.append((line, score))
        return result

    @staticmethod
    def _fallback_line_based(original_code: str) -> List[CodePiece]:
        """Fallback when AST fails: score lines by importance."""
        lines = [l for l in original_code.strip().split('\n') if l.strip()]
        scored = []
        for i, line in enumerate(lines):
            s = 0
            if '"""' in line or "'''" in line or ('"' in line and "'" in line):
                s += 3
            elif '"' in line or "'" in line:
                s += 2
            if 'raise ' in line:
                s += 3
            if 'return ' in line:
                s += 2
            if line.strip().startswith('if ') or line.strip().startswith('elif '):
                s += 2
            if any(c in line for c in '0123456789'):
                s += 1
            scored.append((line.strip(), 100 - i + s * 10))
        scored.sort(key=lambda x: -x[1])
        return [
            CodePiece(code=line, priority=min(99, score), tier='line', description='Line')
            for line, score in scored[:20]
        ]

    @staticmethod
    def get_next_pieces(
        original_code: str,
        already_added: List[str],
        regen_code: str,
        max_pieces_per_iter: int = 2
    ) -> List[CodePiece]:
        """
        Get the next code pieces to add, considering what's already added and
        optionally what's missing in regen (diff-aware).
        """
        all_pieces = SmartCodeExtractor.extract_prioritized_pieces(original_code)
        added_set = {p.strip() for p in already_added}

        # Prefer pieces that appear "missing" in regen (diff-aware)
        if regen_code:
            regen_lower = regen_code.lower()
            def score_missing(p: CodePiece) -> int:
                base = p.priority
                # If this piece's content isn't in regen, boost it
                key = p.code.strip()[:60]
                if key and key not in regen_code and key.lower() not in regen_lower:
                    base += 15
                return base
            all_pieces.sort(key=lambda x: -score_missing(x))
        else:
            all_pieces.sort(key=lambda x: -x.priority)

        next_pieces = []
        for p in all_pieces:
            if len(next_pieces) >= max_pieces_per_iter:
                break
            norm = p.code.strip()
            if norm and norm not in added_set:
                next_pieces.append(p)
                added_set.add(norm)

        return next_pieces
