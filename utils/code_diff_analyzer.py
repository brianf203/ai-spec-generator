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
