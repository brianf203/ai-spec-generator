"""
Utility to analyze code differences and suggest code pieces to add to specs
"""
import ast
import difflib
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


class CodeDiffAnalyzer:
    """Analyzes differences between original and regenerated code"""
    
    @staticmethod
    def get_code_differences(original_code: str, regenerated_code: str) -> Dict[str, Any]:
        """
        Analyze differences between original and regenerated code.
        Returns a dictionary with:
        - missing_lines: lines present in original but not in regenerated
        - extra_lines: lines present in regenerated but not in original
        - diff_blocks: blocks of code that differ
        - suggestions: suggested code pieces to add
        """
        if not original_code or not regenerated_code:
            return {
                'missing_lines': [],
                'extra_lines': [],
                'diff_blocks': [],
                'suggestions': []
            }
        
        original_lines = original_code.split('\n')
        regenerated_lines = regenerated_code.split('\n')
        
        # Use difflib to find differences
        diff = list(difflib.unified_diff(
            regenerated_lines,
            original_lines,
            lineterm='',
            n=0  # Context lines
        ))
        
        missing_lines = []
        extra_lines = []
        diff_blocks = []
        current_block = None
        
        for line in diff:
            if line.startswith('+++') or line.startswith('---'):
                continue
            elif line.startswith('@@'):
                if current_block:
                    diff_blocks.append(current_block)
                current_block = {
                    'type': 'diff',
                    'original': [],
                    'regenerated': []
                }
            elif line.startswith('+'):
                missing_lines.append(line[1:])
                if current_block:
                    current_block['original'].append(line[1:])
            elif line.startswith('-'):
                extra_lines.append(line[1:])
                if current_block:
                    current_block['regenerated'].append(line[1:])
            elif current_block and line.strip():
                if current_block['original'] or current_block['regenerated']:
                    diff_blocks.append(current_block)
                    current_block = {
                        'type': 'diff',
                        'original': [],
                        'regenerated': []
                    }
        
        if current_block and (current_block['original'] or current_block['regenerated']):
            diff_blocks.append(current_block)
        
        # Generate suggestions
        suggestions = CodeDiffAnalyzer._generate_suggestions(
            missing_lines, diff_blocks, original_code, regenerated_code
        )
        
        return {
            'missing_lines': missing_lines,
            'extra_lines': extra_lines,
            'diff_blocks': diff_blocks,
            'suggestions': suggestions
        }
    
    @staticmethod
    def _generate_suggestions(
        missing_lines: List[str],
        diff_blocks: List[Dict[str, Any]],
        original_code: str,
        regenerated_code: str
    ) -> List[Dict[str, Any]]:
        """Generate suggestions for code pieces to add"""
        suggestions = []
        
        # Try to identify meaningful code blocks
        try:
            orig_tree = ast.parse(original_code)
            regen_tree = ast.parse(regenerated_code) if regenerated_code else None
            
            if regen_tree:
                # Compare AST nodes
                orig_nodes = list(ast.walk(orig_tree))
                regen_nodes = list(ast.walk(regen_tree))
                
                # Find missing statements/expressions
                def node_to_str(node):
                    if hasattr(ast, 'unparse'):
                        try:
                            return ast.unparse(node)
                        except:
                            return str(node)
                    return str(node)
                
                orig_statements = [
                    node_to_str(node)
                    for node in orig_nodes
                    if isinstance(node, (ast.Assign, ast.Return, ast.If, ast.For, ast.While, ast.Expr))
                ]
                
                regen_statements = [
                    node_to_str(node)
                    for node in regen_nodes
                    if isinstance(node, (ast.Assign, ast.Return, ast.If, ast.For, ast.While, ast.Expr))
                ]
                
                for stmt in orig_statements:
                    if stmt not in regen_statements:
                        suggestions.append({
                            'type': 'statement',
                            'code': stmt,
                            'priority': 'medium',
                            'description': f'Missing statement: {stmt[:50]}...'
                        })
        except SyntaxError:
            # Fallback to line-based suggestions
            for i, line in enumerate(missing_lines[:10]):  # Limit to first 10
                if line.strip() and not line.strip().startswith('#'):
                    suggestions.append({
                        'type': 'line',
                        'code': line,
                        'priority': 'low',
                        'description': f'Missing line: {line[:50]}...'
                    })
        
        return suggestions
    
    @staticmethod
    def extract_code_piece(original_code: str, line_numbers: List[int]) -> str:
        """Extract a specific code piece by line numbers"""
        lines = original_code.split('\n')
        selected_lines = []
        for line_num in sorted(line_numbers):
            if 0 <= line_num < len(lines):
                selected_lines.append(lines[line_num])
        return '\n'.join(selected_lines)
    
    @staticmethod
    def get_diff_natural_language_descriptions(
        original_code: str,
        regenerated_code: str,
        max_descriptions: int = 15
    ) -> List[str]:
        """
        Produce natural language descriptions of what differs between original and regenerated code.
        Used by the refinement loop to improve prompts without adding code.
        """
        if not original_code or not regenerated_code:
            return []
        diff_result = CodeDiffAnalyzer.get_code_differences(original_code, regenerated_code)
        missing_lines = diff_result.get('missing_lines', [])
        extra_lines = diff_result.get('extra_lines', [])
        diff_blocks = diff_result.get('diff_blocks', [])
        descriptions = []

        # Describe missing content (in original but not in regen)
        for line in missing_lines[:max_descriptions]:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            if 'raise ' in stripped and ('"' in stripped or "'" in stripped):
                # Extract error message if possible
                msg_extracted = False
                for q in ['"', "'"]:
                    if q in stripped:
                        try:
                            start = stripped.index(q) + 1
                            end = stripped.index(q, start)
                            msg = stripped[start:end]
                            descriptions.append(
                                f"Original raises with exact message '{msg}'; regenerated may use different wording or omit it."
                            )
                            msg_extracted = True
                            break
                        except ValueError:
                            pass
                if not msg_extracted:
                    descriptions.append(f"Original has: {stripped[:80]}; regenerated omits or differs.")
            elif stripped.startswith('if ') or stripped.startswith('elif '):
                descriptions.append(
                    f"Original has condition: {stripped[:70]}; regenerated may omit or use different logic."
                )
            elif 'return ' in stripped:
                descriptions.append(
                    f"Original returns: {stripped[:70]}; regenerated may return differently."
                )
            elif '=' in stripped and not stripped.startswith('=='):
                var_part = stripped.split('=')[0].strip()
                descriptions.append(
                    f"Original assigns to {var_part}; regenerated may use different variable or omit."
                )
            else:
                descriptions.append(f"Original has: {stripped[:80]}; regenerated omits or differs.")

        # Describe diff blocks (side-by-side differences)
        for i, block in enumerate(diff_blocks[:5]):
            orig_lines = block.get('original', [])
            regen_lines = block.get('regenerated', [])
            if orig_lines:
                orig_preview = orig_lines[0][:60] if orig_lines else ''
                regen_preview = regen_lines[0][:60] if regen_lines else '(omitted)'
                descriptions.append(
                    f"Diff block {i+1}: Original has '{orig_preview}...' but regenerated has '{regen_preview}...'"
                )

        return descriptions[:max_descriptions]

    @staticmethod
    def _is_docstring_piece(code: str) -> bool:
        """Check if piece is a docstring (triple-quoted or quoted string)."""
        s = code.strip()
        if (s.startswith('"""') and s.endswith('"""')) or (s.startswith("'''") and s.endswith("'''")):
            return True
        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            return len(s) > 1 and '"""' not in s and "'''" not in s
        return False

    @staticmethod
    def _docstring_content(s: str) -> str:
        """Extract inner content from docstring for dedup."""
        s = s.strip()
        for q in ['"""', "'''", '"', "'"]:
            if s.startswith(q) and s.endswith(q) and len(s) > len(q) * 2:
                return s[len(q):-len(q)].strip()
        return s

    @staticmethod
    def _hybrid_line_priority(stripped: str) -> int:
        """
        Higher = add first. Prefer lines that constrain structure/behavior most per token.
        """
        if not stripped or stripped.startswith('#'):
            return 0
        p = 40
        if CodeDiffAnalyzer._is_docstring_piece(stripped):
            p = 100
        elif stripped.startswith(('except ', 'finally:')):
            p = 88
        elif 'raise ' in stripped:
            p = 86
        elif stripped.startswith(('if ', 'elif ', 'while ', 'for ', 'try:', 'with ')):
            p = 78
        elif 'return ' in stripped:
            p = 72
        elif stripped.startswith(('assert ', 'yield ')):
            p = 68
        elif any(q in stripped for q in ('"', "'")):
            p = 65
        elif '=' in stripped and not stripped.startswith('=='):
            p = 45
        return p

    @staticmethod
    def _normalize_line_key(line: str) -> str:
        return ' '.join(line.strip().split())

    @staticmethod
    def _extract_distinctive_literal_snippets(line: str, regenerated_code: str) -> Optional[str]:
        """
        If line is long but contains string literals missing from regenerated code,
        return a minimal hint (quoted strings only) instead of the full line.
        """
        try:
            tree = ast.parse(line)
        except SyntaxError:
            return None
        literals: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.strip():
                s = node.value
                if len(s) >= 2 and s not in regenerated_code:
                    literals.append(repr(s))
        if not literals:
            return None
        if len(literals) == 1:
            return f"# Required string / message: {literals[0]}"
        return "# Required strings: " + ", ".join(literals[:4])

    @staticmethod
    def _extract_minimal_piece(full_stmt: str) -> str:
        """
        Extract smallest differing part from a multi-line statement.
        Prefer single lines; for long blocks return the first distinctive line.
        """
        lines = [l.strip() for l in full_stmt.split('\n') if l.strip()]
        if len(lines) <= 1:
            return full_stmt.strip()
        # Prefer high-impact single lines: docstring, raise, return, condition
        for line in lines:
            if CodeDiffAnalyzer._is_docstring_piece(line) or 'raise ' in line or line.startswith('return '):
                return line
            if line.startswith('if ') or line.startswith('elif ') or line.startswith('while '):
                return line
        return lines[0]

    @staticmethod
    def get_diff_driven_pieces(
        original_code: str,
        regenerated_code: str,
        already_added: List[str],
        max_pieces: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Get MINIMAL code pieces to add - smallest differing part, not whole statements.
        Deduplicates docstrings (add only triple-quoted form once).
        """
        if not original_code or not regenerated_code:
            return []
        diff_result = CodeDiffAnalyzer.get_code_differences(original_code, regenerated_code)
        suggestions = diff_result.get('suggestions', [])
        missing_lines = diff_result.get('missing_lines', [])
        # Dedupe and sort by structural priority (highest first) — was diff-order which wastes iterations
        seen_norm: set = set()
        unique_missing: List[str] = []
        for line in missing_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            nk = CodeDiffAnalyzer._normalize_line_key(stripped)
            if nk in seen_norm:
                continue
            seen_norm.add(nk)
            unique_missing.append(line)
        missing_lines = sorted(
            unique_missing,
            key=lambda ln: -CodeDiffAnalyzer._hybrid_line_priority(ln.strip()),
        )
        added_set = {p.strip() for p in already_added}
        added_docstring_contents = set()

        def _add_piece(code: str, desc: str, priority: int) -> bool:
            if not code or code in added_set:
                return False
            # Deduplicate docstrings: add only triple-quoted form, skip if content already added
            if CodeDiffAnalyzer._is_docstring_piece(code):
                content = CodeDiffAnalyzer._docstring_content(code)
                if content in added_docstring_contents:
                    return False
                # Prefer triple-quoted form
                if not (code.startswith('"""') or code.startswith("'''")):
                    code = f'"""{content}"""'
                added_docstring_contents.add(content)
            pieces.append({'code': code, 'description': desc, 'priority': priority})
            added_set.add(code)
            return True

        pieces = []

        # 1. Prefer missing_lines (priority-sorted) — try literal-only hint for long assignment lines
        for line in missing_lines:
            if len(pieces) >= max_pieces:
                break
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            priority = CodeDiffAnalyzer._hybrid_line_priority(stripped)
            code_to_use = stripped
            if len(stripped) > 100 and '=' in stripped:
                lit_hint = CodeDiffAnalyzer._extract_distinctive_literal_snippets(stripped, regenerated_code)
                if lit_hint:
                    code_to_use = lit_hint
                    priority = max(priority, 80)
            _add_piece(code_to_use, f'Missing: {stripped[:50]}...', priority)

        # 2. If we need more, use suggestions but extract MINIMAL piece (not whole statement)
        if len(pieces) < max_pieces:
            for s in suggestions:
                if len(pieces) >= max_pieces:
                    break
                full_code = s.get('code', '').strip()
                if not full_code or full_code in added_set:
                    continue
                # For multi-line statements, add only the minimal part
                orig_lines = original_code.split('\n')
                minimal = CodeDiffAnalyzer._extract_minimal_piece(full_code)
                if minimal != full_code and len(minimal) < len(full_code):
                    code_to_add = minimal
                else:
                    code_to_add = full_code
                desc = s.get('description', 'Missing from regenerated')
                priority = 90 if s.get('type') == 'statement' else 70
                _add_piece(code_to_add, desc, priority)

        return sorted(pieces, key=lambda p: -p['priority'])

    @staticmethod
    def find_similar_code_blocks(original_code: str, regenerated_code: str) -> List[Dict[str, Any]]:
        """Find similar code blocks that might need to be added"""
        try:
            orig_tree = ast.parse(original_code)
            regen_tree = ast.parse(regenerated_code) if regenerated_code else None
            
            if not regen_tree:
                return []
            
            similar_blocks = []
            
            # Compare function bodies
            orig_funcs = [n for n in ast.walk(orig_tree) if isinstance(n, ast.FunctionDef)]
            regen_funcs = [n for n in ast.walk(regen_tree) if isinstance(n, ast.FunctionDef)]
            
            if orig_funcs and regen_funcs:
                orig_func = orig_funcs[0]
                regen_func = regen_funcs[0]
                
                # Compare body statements
                def stmt_to_str(stmt):
                    if hasattr(ast, 'unparse'):
                        try:
                            return ast.unparse(stmt)
                        except:
                            return str(stmt)
                    return str(stmt)
                
                orig_body = [stmt_to_str(stmt) for stmt in orig_func.body]
                regen_body = [stmt_to_str(stmt) for stmt in regen_func.body]
                
                for i, stmt in enumerate(orig_body):
                    if i >= len(regen_body) or stmt != regen_body[i]:
                        similar_blocks.append({
                            'position': i,
                            'original': stmt,
                            'regenerated': regen_body[i] if i < len(regen_body) else None,
                            'code': stmt
                        })
            
            return similar_blocks
        except SyntaxError:
            return []
