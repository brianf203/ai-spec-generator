"""
Divide and Conquer Algorithm for Complex Function Analysis
Implements algorithmic breakdown of complex functions into manageable chunks
"""

import ast
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict


class DivideAndConquerAnalyzer:
    """Algorithmically breaks down complex functions into manageable chunks"""
    
    def __init__(self):
        self.path_id_counter = 0
    
    def analyze_function_paths(self, code: str) -> Dict[str, Any]:
        """Analyze function and identify distinct execution paths"""
        try:
            tree = ast.parse(code)
            func_node = None
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_node = node
                    break
            
            if not func_node:
                return {'paths': [], 'complexity': 0}
            
            paths = self._extract_execution_paths(func_node)
            path_conditions = self._analyze_path_conditions(func_node, paths)
            
            return {
                'paths': paths,
                'path_conditions': path_conditions,
                'complexity': len(paths),
                'has_nested_loops': self._has_deeply_nested_loops(func_node),
                'has_multiple_returns': self._count_returns(func_node) > 1,
                'branching_factor': self._calculate_branching_factor(func_node)
            }
        except Exception:
            return {'paths': [], 'complexity': 0}
    
    def _extract_execution_paths(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract distinct execution paths through the function"""
        paths = []
        self.path_id_counter = 0
        visited_paths = set()
        
        def traverse_path(node: Optional[ast.AST], current_path: List[ast.AST], conditions: List[str], depth: int = 0):
            if depth > 25 or node is None:
                return
            
            if isinstance(node, list):
                for n in node:
                    traverse_path(n, current_path.copy(), conditions.copy(), depth)
                return
            
            current_path.append(node)
            path_signature = (tuple(conditions), type(node).__name__)
            
            if path_signature in visited_paths and depth > 5:
                return
            visited_paths.add(path_signature)
            
            if isinstance(node, ast.If):
                true_condition = self._extract_condition(node.test)
                true_conditions = conditions + [f"if {true_condition}"]
                false_conditions = conditions + [f"else (not {true_condition})"]
                
                if node.body:
                    for stmt in node.body:
                        traverse_path(stmt, current_path.copy(), true_conditions, depth + 1)
                
                if node.orelse:
                    for stmt in node.orelse:
                        traverse_path(stmt, current_path.copy(), false_conditions, depth + 1)
                else:
                    paths.append({
                        'id': f"path_{self.path_id_counter}",
                        'nodes': [n for n in current_path if n],
                        'conditions': false_conditions,
                        'type': 'conditional_else',
                        'line': getattr(node, 'lineno', None)
                    })
                    self.path_id_counter += 1
            elif isinstance(node, ast.While):
                loop_condition = self._extract_condition(node.test)
                loop_conditions = conditions + [f"while {loop_condition}"]
                
                if node.body:
                    for stmt in node.body:
                        traverse_path(stmt, current_path.copy(), loop_conditions, depth + 1)
                
                if node.orelse:
                    for stmt in node.orelse:
                        traverse_path(stmt, current_path.copy(), conditions + [f"while else"], depth + 1)
            elif isinstance(node, ast.For):
                loop_conditions = conditions + ["for loop iteration"]
                
                if node.body:
                    for stmt in node.body:
                        traverse_path(stmt, current_path.copy(), loop_conditions, depth + 1)
                
                if node.orelse:
                    for stmt in node.orelse:
                        traverse_path(stmt, current_path.copy(), conditions + [f"for else"], depth + 1)
            elif isinstance(node, ast.Try):
                try_conditions = conditions + ["try block"]
                except_conditions = conditions + ["except block"]
                else_conditions = conditions + ["try else"]
                finally_conditions = conditions + ["finally block"]
                
                if node.body:
                    for stmt in node.body:
                        traverse_path(stmt, current_path.copy(), try_conditions, depth + 1)
                
                for handler in node.handlers:
                    if handler.body:
                        for stmt in handler.body:
                            traverse_path(stmt, current_path.copy(), except_conditions, depth + 1)
                
                if node.orelse:
                    for stmt in node.orelse:
                        traverse_path(stmt, current_path.copy(), else_conditions, depth + 1)
                
                if node.finalbody:
                    for stmt in node.finalbody:
                        traverse_path(stmt, current_path.copy(), finally_conditions, depth + 1)
            elif isinstance(node, ast.Return):
                paths.append({
                    'id': f"path_{self.path_id_counter}",
                    'nodes': [n for n in current_path if n],
                    'conditions': conditions.copy(),
                    'type': 'return_path',
                    'return_value': self._extract_condition(node.value) if node.value else None,
                    'line': getattr(node, 'lineno', None)
                })
                self.path_id_counter += 1
            elif isinstance(node, ast.Raise):
                paths.append({
                    'id': f"path_{self.path_id_counter}",
                    'nodes': [n for n in current_path if n],
                    'conditions': conditions.copy(),
                    'type': 'exception_path',
                    'exception': self._extract_condition(node.exc) if node.exc else "Exception",
                    'line': getattr(node, 'lineno', None)
                })
                self.path_id_counter += 1
            elif hasattr(node, 'body') and node.body:
                for child in node.body:
                    traverse_path(child, current_path.copy(), conditions.copy(), depth + 1)
        
        if func_node.body:
            for stmt in func_node.body:
                traverse_path(stmt, [], [], 0)
        
        if not paths:
            paths.append({
                'id': 'path_0',
                'nodes': list(func_node.body) if func_node.body else [],
                'conditions': [],
                'type': 'linear',
                'line': func_node.lineno if hasattr(func_node, 'lineno') else None
            })
        
        return paths
    
    def _extract_condition(self, node: ast.AST) -> str:
        """Extract condition as string"""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(node)
            elif isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Compare):
                return f"{self._extract_condition(node.left)} {[op.__class__.__name__ for op in node.ops]} {[self._extract_condition(c) for c in node.comparators]}"
            else:
                return str(type(node).__name__)
        except Exception:
            return str(type(node).__name__)
    
    def _analyze_path_conditions(self, func_node: ast.FunctionDef, paths: List[Dict]) -> Dict[str, List[str]]:
        """Analyze conditions that lead to each path"""
        path_conditions = {}
        
        for path in paths:
            conditions = path.get('conditions', [])
            path_id = path['id']
            path_conditions[path_id] = conditions
        
        return path_conditions
    
    def _has_deeply_nested_loops(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has deeply nested loops"""
        max_depth = 0
        
        def check_depth(node: ast.AST, depth: int = 0):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            if isinstance(node, (ast.For, ast.While)):
                if hasattr(node, 'body'):
                    for child in node.body:
                        check_depth(child, depth + 1)
            elif hasattr(node, 'body'):
                for child in node.body:
                    check_depth(child, depth)
        
        if func_node.body:
            for stmt in func_node.body:
                check_depth(stmt, 0)
        
        return max_depth >= 3
    
    def _count_returns(self, func_node: ast.FunctionDef) -> int:
        """Count number of return statements"""
        count = 0
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return):
                count += 1
        return count
    
    def _calculate_branching_factor(self, func_node: ast.FunctionDef) -> int:
        """Calculate branching factor (number of decision points)"""
        branches = 0
        for node in ast.walk(func_node):
            if isinstance(node, ast.If):
                branches += 1
            elif isinstance(node, ast.BoolOp):
                branches += len(node.values) - 1
        return branches
    
    def generate_path_specification_prompt(self, code: str, path: Dict[str, Any], all_paths: List[Dict]) -> str:
        """Generate a focused specification prompt for a specific path"""
        path_id = path['id']
        conditions = path.get('conditions', [])
        path_type = path.get('type', 'unknown')
        
        conditions_text = " AND ".join(conditions) if conditions else "default/linear path"
        
        prompt = f"""
Analyze this SPECIFIC execution path of a complex function. This is one of {len(all_paths)} distinct paths.

Function code:
```python
{code}
```

FOCUS ON THIS PATH ONLY:
- Path ID: {path_id}
- Path Type: {path_type}
- Conditions: {conditions_text}
- This path executes when: {conditions_text}

Generate a detailed specification for THIS PATH ONLY, including:
1. The exact conditions that trigger this path
2. The sequence of operations in this path
3. Variable assignments and modifications in this path
4. Return value or side effects for this path
5. Any error handling specific to this path

Be extremely detailed about this specific path. Ignore other paths for now.
"""
        return prompt
    
    def merge_path_specifications(self, path_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge specifications from multiple paths into a unified spec with intelligent deduplication"""
        if not path_specs:
            return {}
        
        merged = {
            'user_stories': [],
            'test_matrix': [],
            'edge_cases': [],
            'internal_logic': []
        }
        
        seen_conditions = set()
        path_priorities = {'return_path': 'P1', 'exception_path': 'P1', 'conditional_else': 'P2', 'loop_body': 'P2', 'linear': 'P3'}
        
        for i, path_spec in enumerate(path_specs):
            path_id = path_spec.get('path_id', f'path_{i}')
            path_type = path_spec.get('type', 'unknown')
            conditions = tuple(path_spec.get('conditions', []))
            
            # Skip duplicate paths with same conditions
            if conditions in seen_conditions and len(conditions) > 0:
                continue
            seen_conditions.add(conditions)
            
            priority = path_priorities.get(path_type, 'P2')
            conditions_text = ', '.join(path_spec.get('conditions', [])) if path_spec.get('conditions') else 'default path'
            
            story = {
                'id': f"US-PATH-{i+1:02d}",
                'priority': priority,
                'title': f"Execution path {i+1}: {path_type}",
                'narrative': f"This path executes when: {conditions_text}. Path type: {path_type}.",
                'acceptance': [{
                    'given': conditions_text or 'default conditions',
                    'when': 'the function executes',
                    'then': f"it follows the {path_type} behavior pattern"
                }]
            }
            
            if path_spec.get('return_value'):
                story['acceptance'][0]['then'] += f" and returns {path_spec['return_value']}"
            if path_spec.get('exception'):
                story['acceptance'][0]['then'] += f" and raises {path_spec['exception']}"
            
            merged['user_stories'].append(story)
            
            if path_spec.get('test_cases'):
                for test in path_spec['test_cases']:
                    test['story_refs'] = [story['id']]
                    merged['test_matrix'].append(test)
            
            if path_spec.get('edge_cases'):
                for edge_case in path_spec['edge_cases']:
                    if isinstance(edge_case, dict):
                        edge_case['story_ref'] = story['id']
                    merged['edge_cases'].append(edge_case)
            
            if path_spec.get('logic'):
                merged['internal_logic'].append(f"Path {i+1} ({path_type}): {path_spec['logic']}")
        
        return merged


class DeltaImprovementAlgorithm:
    """Delta improvement algorithm that focuses on specific similarity gaps"""
    
    def __init__(self):
        self.improvement_history = []
    
    def identify_improvement_deltas(self, similarity_metrics: Dict[str, float], 
                                   test_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific areas that need improvement (deltas) with priority scoring"""
        deltas = []
        
        structural = similarity_metrics.get('structural_similarity', 0.0)
        behavioral = similarity_metrics.get('behavioral_similarity', 0.0)
        behavioral_test = similarity_metrics.get('behavioral_test_similarity', 0.0)
        coverage = test_results.get('branch_coverage', 0.0)
        
        # Calculate priority scores (higher = more urgent)
        def calculate_priority_score(current: float, target: float, impact_weight: float = 1.0) -> float:
            gap = target - current
            return gap * impact_weight
        
        if structural < 0.8:
            priority_score = calculate_priority_score(structural, 0.95, 1.2)
            deltas.append({
                'type': 'structural',
                'current': structural,
                'target': 0.95,
                'priority': 'high' if structural < 0.7 else 'medium',
                'priority_score': priority_score,
                'focus': 'code structure, variable names, control flow organization, AST node types'
            })
        
        if behavioral < 0.8:
            priority_score = calculate_priority_score(behavioral, 0.95, 1.1)
            deltas.append({
                'type': 'behavioral',
                'current': behavioral,
                'target': 0.95,
                'priority': 'high' if behavioral < 0.7 else 'medium',
                'priority_score': priority_score,
                'focus': 'function behavior patterns, logic flow, algorithm correctness'
            })
        
        if behavioral_test < 0.8:
            priority_score = calculate_priority_score(behavioral_test, 0.95, 1.3)
            deltas.append({
                'type': 'behavioral_test',
                'current': behavioral_test,
                'target': 0.95,
                'priority': 'high' if behavioral_test < 0.6 else 'medium',
                'priority_score': priority_score,
                'focus': 'test pass rate, output correctness, exception handling'
            })
        
        if coverage < 0.8:
            missing_lines = test_results.get('missing_lines', [])
            missing_branches = test_results.get('missing_branches', [])
            priority_score = calculate_priority_score(coverage, 0.95, 1.4)
            deltas.append({
                'type': 'coverage',
                'current': coverage,
                'target': 0.95,
                'priority': 'high' if coverage < 0.6 else 'medium',
                'priority_score': priority_score,
                'focus': f'missing lines: {missing_lines[:10]}, missing branches: {len(missing_branches)}',
                'missing_lines': missing_lines,
                'missing_branches': missing_branches
            })
        
        # Sort by priority score (highest first), then by current value (lowest first)
        return sorted(deltas, key=lambda x: (-x.get('priority_score', 0), x['current']))
    
    def generate_delta_focused_prompt(self, base_prompt: str, deltas: List[Dict[str, Any]], 
                                     iteration: int) -> str:
        """Generate a prompt focused on specific improvement deltas with actionable guidance"""
        if not deltas:
            return base_prompt
        
        delta_section = "\n\n" + "="*70 + "\n"
        delta_section += f"CRITICAL IMPROVEMENT FOCUS (Iteration {iteration}):\n"
        delta_section += "="*70 + "\n"
        delta_section += "The following gaps were identified. Address them with HIGH PRIORITY:\n\n"
        
        for i, delta in enumerate(deltas[:4], 1):
            priority_emoji = "🔴" if delta.get('priority') == 'high' else "🟡"
            delta_section += f"{priority_emoji} {i}. {delta['type'].upper().replace('_', ' ')} GAP (Priority: {delta.get('priority', 'medium').upper()}):\n"
            delta_section += f"   Current: {delta['current']:.1%} → Target: {delta['target']:.1%} (Gap: {delta['target'] - delta['current']:.1%})\n"
            delta_section += f"   Action Required: {delta['focus']}\n"
            
            if 'missing_lines' in delta and delta['missing_lines']:
                delta_section += f"   Specific missing lines to cover: {delta['missing_lines'][:8]}\n"
            if 'missing_branches' in delta and delta['missing_branches']:
                branch_count = len(delta['missing_branches'])
                delta_section += f"   Missing branches to exercise: {branch_count} branch(es)\n"
            
            # Add type-specific guidance
            if delta['type'] == 'structural':
                delta_section += "   → Ensure variable names, control flow structure, and AST node types match exactly\n"
            elif delta['type'] == 'behavioral':
                delta_section += "   → Verify logic flow, algorithm steps, and data transformations match\n"
            elif delta['type'] == 'behavioral_test':
                delta_section += "   → Fix test failures: ensure outputs and exceptions match original behavior\n"
            elif delta['type'] == 'coverage':
                delta_section += "   → Generate tests that execute missing lines and branches\n"
            
            delta_section += "\n"
        
        delta_section += "="*70 + "\n"
        delta_section += "Generate specifications/tests that specifically address these gaps in priority order.\n"
        delta_section += "Focus on the highest priority gaps first.\n"
        
        return base_prompt + delta_section
    
    def should_use_divide_conquer(self, complexity_metrics: Dict[str, Any]) -> bool:
        """Determine if divide-and-conquer should be used"""
        complexity = complexity_metrics.get('complexity', 1)
        cyclomatic = complexity_metrics.get('cyclomatic_complexity', complexity)
        num_paths = complexity_metrics.get('num_paths', 0)
        branching_factor = complexity_metrics.get('branching_factor', 0)
        
        return (
            complexity > 10 or
            cyclomatic > 12 or
            num_paths > 5 or
            branching_factor > 6
        )
    
    def calculate_adaptive_iterations(self, complexity_metrics: Dict[str, Any], base_iterations: int = 3) -> int:
        """Calculate adaptive iteration limit based on complexity"""
        complexity = complexity_metrics.get('complexity', 1)
        cyclomatic = complexity_metrics.get('cyclomatic_complexity', complexity)
        num_paths = complexity_metrics.get('num_paths', 0)
        branching_factor = complexity_metrics.get('branching_factor', 0)
        
        iterations = base_iterations
        
        if complexity > 15:
            iterations += 2
        elif complexity > 10:
            iterations += 1
        
        if cyclomatic > 20:
            iterations += 2
        elif cyclomatic > 15:
            iterations += 1
        
        if num_paths > 10:
            iterations += 2
        elif num_paths > 5:
            iterations += 1
        
        if branching_factor > 10:
            iterations += 1
        
        return min(iterations, 10)

