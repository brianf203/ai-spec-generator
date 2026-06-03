"""
Advanced Analysis Agents
Specialized agents for different types of code analysis
"""

import ast
import json
import os
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
import networkx as nx
import numpy as np


class AdvancedCodeAnalyzer:
    """Advanced code analysis with pattern recognition"""
    
    def __init__(self):
        self.patterns = {
            'design_patterns': self._detect_design_patterns,
            'algorithm_patterns': self._detect_algorithm_patterns,
            'data_flow': self._analyze_data_flow,
            'control_flow': self._analyze_control_flow,
            'dependency_graph': self._build_dependency_graph
        }
    
    def analyze_code_advanced(self, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform advanced code analysis"""
        try:
            tree = ast.parse(code)
            analysis = {
                'ast_structure': self._analyze_ast_structure(tree),
                'design_patterns': self._detect_design_patterns(tree),
                'algorithm_patterns': self._detect_algorithm_patterns(tree),
                'data_flow': self._analyze_data_flow(tree),
                'control_flow': self._analyze_control_flow(tree),
                'complexity_metrics': self._calculate_advanced_complexity(tree),
                'semantic_features': self._extract_semantic_features(tree),
                'dependencies': self._extract_dependencies(tree),
                'variable_usage': self._analyze_variable_usage(tree),
                'function_calls': self._analyze_function_calls(tree)
            }
            
            return analysis
            
        except SyntaxError as e:
            return {'error': f'Syntax error: {e}'}
    
    def _analyze_ast_structure(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze AST structure in detail"""
        structure = {
            'node_types': Counter(),
            'nesting_depth': 0,
            'branching_factor': 0,
            'leaf_nodes': 0
        }
        
        def analyze_node(node, depth=0):
            structure['node_types'][type(node).__name__] += 1
            structure['nesting_depth'] = max(structure['nesting_depth'], depth)
            
            if hasattr(node, 'body'):
                structure['branching_factor'] += len(node.body)
                for child in node.body:
                    analyze_node(child, depth + 1)
            else:
                structure['leaf_nodes'] += 1
        
        analyze_node(tree)
        return structure
    
    def _detect_design_patterns(self, tree: ast.AST) -> List[str]:
        """Detect common design patterns"""
        patterns = []
        
        # Singleton pattern
        if self._is_singleton(tree):
            patterns.append('singleton')
        
        # Factory pattern
        if self._is_factory(tree):
            patterns.append('factory')
        
        # Observer pattern
        if self._is_observer(tree):
            patterns.append('observer')
        
        # Decorator pattern
        if self._is_decorator(tree):
            patterns.append('decorator')
        
        return patterns
    
    def _detect_algorithm_patterns(self, tree: ast.AST) -> List[str]:
        """Detect common algorithm patterns"""
        patterns = []
        
        # Recursive algorithms
        if self._has_recursion(tree):
            patterns.append('recursive')
        
        # Dynamic programming
        if self._has_memoization(tree):
            patterns.append('dynamic_programming')
        
        # Divide and conquer
        if self._has_divide_conquer(tree):
            patterns.append('divide_conquer')
        
        # Greedy algorithms
        if self._has_greedy(tree):
            patterns.append('greedy')
        
        return patterns
    
    def _analyze_data_flow(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze data flow through the code"""
        variables = set()
        assignments = []
        usages = []
        data_dependencies = defaultdict(list)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.add(target.id)
                        assignments.append({
                            'variable': target.id,
                            'line': node.lineno,
                            'value': ast.unparse(node.value) if hasattr(ast, 'unparse') else str(node.value)
                        })
            
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                usages.append({
                    'variable': node.id,
                    'line': node.lineno
                })
        
        # Convert sets to lists for JSON serialization
        data_flow = {
            'variables': list(variables),
            'assignments': assignments,
            'usages': usages,
            'data_dependencies': dict(data_dependencies)
        }
        
        return data_flow
    
    def _analyze_control_flow(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze control flow patterns"""
        control_flow = {
            'if_statements': 0,
            'loops': 0,
            'try_blocks': 0,
            'with_statements': 0,
            'nested_depth': 0,
            'branching_points': []
        }
        
        def analyze_control(node, depth=0):
            if isinstance(node, ast.If):
                control_flow['if_statements'] += 1
                control_flow['branching_points'].append({
                    'type': 'if',
                    'line': node.lineno,
                    'depth': depth
                })
            elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                control_flow['loops'] += 1
                control_flow['branching_points'].append({
                    'type': 'loop',
                    'line': node.lineno,
                    'depth': depth
                })
            elif isinstance(node, ast.Try):
                control_flow['try_blocks'] += 1
            elif isinstance(node, ast.With):
                control_flow['with_statements'] += 1
            
            control_flow['nested_depth'] = max(control_flow['nested_depth'], depth)
            
            if hasattr(node, 'body'):
                for child in node.body:
                    analyze_control(child, depth + 1)
        
        analyze_control(tree)
        return control_flow
    
    def _calculate_advanced_complexity(self, tree: ast.AST) -> Dict[str, float]:
        """Calculate advanced complexity metrics"""
        metrics = {
            'cyclomatic_complexity': 1,
            'cognitive_complexity': 0,
            'nesting_complexity': 0,
            'halstead_volume': 0
        }
        
        # Cyclomatic complexity
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler)):
                metrics['cyclomatic_complexity'] += 1
            elif isinstance(node, ast.BoolOp):
                metrics['cyclomatic_complexity'] += len(node.values) - 1
        
        # Cognitive complexity (simplified)
        nesting_level = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                nesting_level += 1
                metrics['cognitive_complexity'] += nesting_level
            elif isinstance(node, ast.BoolOp):
                metrics['cognitive_complexity'] += len(node.values) - 1
        
        return metrics
    
    def _extract_semantic_features(self, tree: ast.AST) -> Dict[str, Any]:
        """Extract semantic features from code"""
        keywords = set()
        operators = set()
        literals = set()
        identifiers = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.Constant):
                literals.add(str(node.value))
            elif isinstance(node, ast.BinOp):
                operators.add(type(node.op).__name__)
            elif isinstance(node, ast.Compare):
                operators.add(type(node.ops[0]).__name__)
        
        # Calculate semantic density
        total_nodes = len(list(ast.walk(tree)))
        semantic_nodes = len(identifiers) + len(operators)
        semantic_density = semantic_nodes / max(total_nodes, 1)
        
        # Convert sets to lists for JSON serialization
        features = {
            'keywords': list(keywords),
            'operators': list(operators),
            'literals': list(literals),
            'identifiers': list(identifiers),
            'semantic_density': semantic_density
        }
        
        return features
    
    def _extract_dependencies(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Extract code dependencies"""
        dependencies = {
            'imports': [],
            'function_calls': [],
            'class_references': [],
            'module_references': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        dependencies['imports'].append(f"import {alias.name} as {alias.asname}")
                    else:
                        dependencies['imports'].append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    parts = []
                    for alias in node.names:
                        if alias.asname:
                            parts.append(f"{alias.name} as {alias.asname}")
                        else:
                            parts.append(alias.name)
                    dependencies['imports'].append(f"from {node.module} import {', '.join(parts)}")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    dependencies['function_calls'].append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        dependencies['function_calls'].append(f"{node.func.value.id}.{node.func.attr}")
        
        return dependencies
    
    def _analyze_variable_usage(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze variable usage patterns"""
        usage = {
            'variable_lifecycle': defaultdict(list),
            'scope_depth': defaultdict(int),
            'usage_frequency': Counter(),
            'variable_types': defaultdict(set)
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                var_name = node.id
                usage['usage_frequency'][var_name] += 1
                
                if isinstance(node.ctx, ast.Store):
                    usage['variable_lifecycle'][var_name].append({
                        'action': 'assignment',
                        'line': node.lineno
                    })
                else:
                    usage['variable_lifecycle'][var_name].append({
                        'action': 'usage',
                        'line': node.lineno
                    })
        
        return usage
    
    def _analyze_function_calls(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze function call patterns"""
        calls = {
            'call_graph': defaultdict(list),
            'recursive_calls': [],
            'external_calls': [],
            'internal_calls': []
        }
        
        function_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_names.add(node.name)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in function_names:
                        calls['internal_calls'].append(func_name)
                    else:
                        calls['external_calls'].append(func_name)
        
        return calls
    
    def _build_dependency_graph(self, tree: ast.AST) -> nx.DiGraph:
        """Build dependency graph from AST"""
        graph = nx.DiGraph()
        
        # Add nodes for functions and classes
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                graph.add_node(node.name, type=type(node).__name__)
        
        # Add edges for dependencies
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    # Find the containing function
                    for parent in ast.walk(tree):
                        if isinstance(parent, ast.FunctionDef):
                            if node in ast.walk(parent):
                                graph.add_edge(parent.name, node.func.id)
                                break
        
        return graph
    
    # Pattern detection helper methods
    def _is_singleton(self, tree: ast.AST) -> bool:
        """Check if code implements singleton pattern"""
        # Simplified singleton detection
        class_defs = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        return len(class_defs) == 1 and any('instance' in ast.unparse(node) for node in class_defs if hasattr(ast, 'unparse'))
    
    def _is_factory(self, tree: ast.AST) -> bool:
        """Check if code implements factory pattern"""
        # Look for factory-like patterns
        return any('create' in ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and hasattr(ast, 'unparse'))
    
    def _is_observer(self, tree: ast.AST) -> bool:
        """Check if code implements observer pattern"""
        # Look for observer-like patterns
        return any('notify' in ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and hasattr(ast, 'unparse'))
    
    def _is_decorator(self, tree: ast.AST) -> bool:
        """Check if code implements decorator pattern"""
        # Look for decorator patterns
        return any(len(node.decorator_list) > 0 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
    
    def _has_recursion(self, tree: ast.AST) -> bool:
        """Check if code has recursion"""
        function_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_names.add(node.name)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in function_names:
                    return True
        return False
    
    def _has_memoization(self, tree: ast.AST) -> bool:
        """Check if code has memoization"""
        # Look for memoization patterns
        return any('cache' in ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and hasattr(ast, 'unparse'))
    
    def _has_divide_conquer(self, tree: ast.AST) -> bool:
        """Check if code has divide and conquer pattern"""
        # Look for divide and conquer patterns
        return any('split' in ast.unparse(node) or 'divide' in ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and hasattr(ast, 'unparse'))
    
    def _has_greedy(self, tree: ast.AST) -> bool:
        """Check if code has greedy algorithm pattern"""
        # Look for greedy patterns
        return any('greedy' in ast.unparse(node) or 'optimal' in ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and hasattr(ast, 'unparse'))

    def _normalize_for_comparison(self, tree: ast.AST) -> ast.AST:
        """Normalize AST for comparison (extract function from class if needed)"""
        # If tree has a class with one method, extract just the method
        if isinstance(tree, ast.Module):
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and len(node.body) == 1:
                    method = node.body[0]
                    if isinstance(method, ast.FunctionDef):
                        # Create a module with just the function
                        normalized = ast.Module(body=[method], type_ignores=[])
                        return normalized
        
        return tree


class SemanticSimilarityAnalyzer:
    """Similarity analysis focused on structure and behavioral patterns"""
    
    def __init__(self):
        self.analyzer = AdvancedCodeAnalyzer()

    @staticmethod
    def coerce_python_source_for_ast(raw: Optional[str]) -> str:
        """
        Strip LLM packaging (markdown fences, leading prose) before ``ast.parse``.
        Regenerated snippets often arrive wrapped in ``` fences; without this, structural similarity
        falsely collapses to zero on SyntaxError.
        """
        from textwrap import dedent

        if not raw:
            return ''
        text = raw.strip()
        text = re.sub(
            r"(?m)^\s*```(?:python|py)?\s*\r?\n?",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"(?m)\r?\n?\s*```\s*$", "", text.strip())
        text = dedent(text).strip()

        lines = text.splitlines()
        for i, line in enumerate(lines):
            s = line.lstrip()
            if s.startswith(("def ", "async def ", "class ")):
                j = i
                while j > 0 and lines[j - 1].lstrip().startswith("@"):
                    j -= 1
                return "\n".join(lines[j:]).strip()
        return text or ""
    
    def calculate_semantic_similarity(self, code1: str, code2: str) -> Dict[str, float]:
        """
        Calculate similarity metrics for regenerated code.
        Returns textual, structural, and behavioral similarity metrics.
        """
        return {
            'textual_similarity': self._textual_similarity(code1, code2),
            'structural_similarity': self._structural_similarity(code1, code2),
            'behavioral_similarity': self._behavioral_similarity(code1, code2),
        }
    
    def _textual_similarity(self, code1: str, code2: str) -> float:
        """Calculate textual similarity using normalized string comparison"""
        if not code1 or not code2:
            return 0.0
        try:
            from difflib import SequenceMatcher
            from textwrap import dedent
            
            # Normalize whitespace and indentation
            code1_clean = dedent(code1).strip() or code1.strip()
            code2_clean = dedent(code2).strip() or code2.strip()
            
            # Calculate normalized string similarity
            similarity = SequenceMatcher(None, code1_clean, code2_clean).ratio()
            return similarity
        except Exception as e:
            # Fallback: basic string comparison
            try:
                from difflib import SequenceMatcher
                similarity = SequenceMatcher(None, code1.strip(), code2.strip()).ratio()
                return similarity
            except:
                return 0.0
    
    def _structural_similarity(self, code1: str, code2: str) -> float:
        """Calculate structural similarity using enhanced AST comparison"""
        try:
            # Empty only — never hard-fail valid short stubs (e.g. `pass` is 4 chars but parseable).
            if not code1 or not code2 or not code1.strip() or not code2.strip():
                return 0.0

            code1_dedented = self.coerce_python_source_for_ast(code1)
            code2_dedented = self.coerce_python_source_for_ast(code2)
            if not code1_dedented or not code2_dedented:
                return 0.0

            tree1 = ast.parse(code1_dedented)
            tree2 = ast.parse(code2_dedented)
            
            # Normalize code for class methods (extract just the function/method if wrapped in class)
            tree1 = self.analyzer._normalize_for_comparison(tree1)
            tree2 = self.analyzer._normalize_for_comparison(tree2)
            
            # Extract detailed structural features
            struct1 = self._extract_enhanced_structure(tree1)
            struct2 = self._extract_enhanced_structure(tree2)
            
            # Check if structures are truly empty (not just empty dicts/lists)
            struct1_has_content = bool(struct1 and (struct1.get('node_types') or struct1.get('variables') or 
                                                     struct1.get('signatures') or struct1.get('control_flow')))
            struct2_has_content = bool(struct2 and (struct2.get('node_types') or struct2.get('variables') or 
                                                     struct2.get('signatures') or struct2.get('control_flow')))
            
            # If structures are empty, return 0.0 (no fallback inflation)
            if not struct1_has_content or not struct2_has_content:
                return 0.0
            
            # Calculate multi-dimensional similarity
            similarities = []
            
            # Node type distribution similarity
            type_sim = self._compare_node_type_distributions(struct1['node_types'], struct2['node_types'])
            similarities.append(('node_types', type_sim, 0.25))
            
            # Control flow structure similarity
            control_sim = self._compare_control_flow(struct1['control_flow'], struct2['control_flow'])
            similarities.append(('control_flow', control_sim, 0.30))
            
            # Expression structure similarity
            expr_sim = self._compare_expression_structure(struct1['expressions'], struct2['expressions'])
            similarities.append(('expressions', expr_sim, 0.20))
            
            # Variable usage pattern similarity (increased weight - critical for matching)
            var_sim = self._compare_variable_patterns(struct1['variables'], struct2['variables'])
            similarities.append(('variables', var_sim, 0.20))  # Increased from 0.15
            
            # Function signature similarity
            sig_sim = self._compare_function_signatures(struct1['signatures'], struct2['signatures'])
            similarities.append(('signatures', sig_sim, 0.10))
            
            # Weighted average - NO semantic equivalence boost
            total_weight = sum(weight for _, _, weight in similarities)
            weighted_sum = sum(sim * weight for _, sim, weight in similarities)
            base_similarity = weighted_sum / total_weight if total_weight > 0 else 0.0
            
            # Return actual calculated similarity (no artificial boost)
            return base_similarity
        except SyntaxError:
            # If parsing fails (markdown fences, truncated output), similarity is undefined.
            return 0.0
        except Exception as e:
            # Previously swallowed all errors → scores looked like LLM failures when the metric crashed.
            if os.environ.get("VERBOSE_SIMILARITY"):
                print(f"[SemanticSimilarityAnalyzer] structural error: {e}", flush=True)
            return 0.0
    
    def _check_semantic_equivalence_boost(
        self, code1: str, code2: str, tree1: ast.AST, tree2: ast.AST, base_sim: float
    ) -> float:
        """DEPRECATED: No longer used - semantic boost was inflating scores artificially"""
        # Return 0.0 - no boost applied
        return 0.0
    
    def _get_call_name(self, call_node: ast.Call) -> str:
        """Get function name from call node"""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            return call_node.func.attr
        return ""
    
    def _extract_enhanced_structure(self, tree: ast.AST) -> Dict[str, Any]:
        """Extract enhanced structural features from AST"""
        from collections import Counter
        
        node_types = Counter()
        control_flow = []
        expressions = []
        variables = set()
        signatures = []
        
        def analyze_node(node, depth=0):
            node_type = type(node).__name__
            node_types[node_type] += 1
            
            if isinstance(node, ast.FunctionDef):
                # Extract function signature
                args = [arg.arg for arg in node.args.args]
                sig = {
                    'name': node.name,
                    'args': args,
                    'num_args': len(args),
                    'has_defaults': len(node.args.defaults) > 0,
                    'has_varargs': node.args.vararg is not None,
                    'has_kwargs': node.args.kwarg is not None
                }
                signatures.append(sig)
            
            elif isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.Try)):
                control_flow.append({
                    'type': node_type,
                    'depth': depth,
                    'has_else': hasattr(node, 'orelse') and len(node.orelse) > 0
                })
            
            elif isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call)):
                expressions.append(node_type)
            
            elif isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    variables.add(node.id)
            
            # Recursively analyze children
            for child in ast.iter_child_nodes(node):
                analyze_node(child, depth + 1)
        
        analyze_node(tree)
        
        result = {
            'node_types': dict(node_types),
            'control_flow': control_flow,
            'expressions': expressions,
            'variables': list(variables),
            'signatures': signatures
        }
        
        # For very simple code, ensure we have at least something
        if not any([result['node_types'], result['control_flow'], result['expressions'], 
                    result['variables'], result['signatures']]):
            # Even if empty, add basic structure from tree itself
            result['node_types'] = {type(tree).__name__: 1}
        
        return result
    
    def _compare_node_type_distributions(self, dist1: Dict[str, int], dist2: Dict[str, int]) -> float:
        """Compare distributions of AST node types"""
        all_types = set(dist1.keys()) | set(dist2.keys())
        if not all_types:
            return 1.0
        
        total1 = sum(dist1.values())
        total2 = sum(dist2.values())
        if total1 == 0 and total2 == 0:
            return 1.0
        # If one is empty and other isn't, return low similarity (structural difference)
        if total1 == 0 or total2 == 0:
            return 0.2
        
        # Normalize distributions
        norm1 = {k: v / total1 for k, v in dist1.items()}
        norm2 = {k: v / total2 for k, v in dist2.items()}
        
        # Calculate cosine similarity
        dot_product = sum(norm1.get(k, 0) * norm2.get(k, 0) for k in all_types)
        mag1 = sum(v * v for v in norm1.values()) ** 0.5
        mag2 = sum(v * v for v in norm2.values()) ** 0.5
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def _compare_control_flow(self, flow1: List[Dict], flow2: List[Dict]) -> float:
        """Compare control flow structures"""
        if not flow1 and not flow2:
            return 1.0
        # If one is empty and other isn't, return low similarity (structural difference)
        if not flow1 or not flow2:
            return 0.2
        
        # Compare sequences of control structures
        seq1 = [f"{f['type']}:{f['depth']}" for f in flow1]
        seq2 = [f"{f['type']}:{f['depth']}" for f in flow2]
        
        import difflib
        return difflib.SequenceMatcher(None, seq1, seq2).ratio()
    
    def _compare_expression_structure(self, expr1: List[str], expr2: List[str]) -> float:
        """Compare expression structures"""
        from collections import Counter
        c1 = Counter(expr1)
        c2 = Counter(expr2)
        
        all_exprs = set(c1.keys()) | set(c2.keys())
        if not all_exprs:
            return 1.0
        
        total1 = sum(c1.values())
        total2 = sum(c2.values())
        if total1 == 0 and total2 == 0:
            return 1.0
        # If one is empty and other isn't, return low similarity (structural difference)
        if total1 == 0 or total2 == 0:
            return 0.2
        
        # Jaccard similarity on normalized counts
        norm1 = {k: v / total1 for k, v in c1.items()}
        norm2 = {k: v / total2 for k, v in c2.items()}
        
        intersection = sum(min(norm1.get(k, 0), norm2.get(k, 0)) for k in all_exprs)
        union = sum(max(norm1.get(k, 0), norm2.get(k, 0)) for k in all_exprs)
        
        return intersection / union if union > 0 else 0.0
    
    def _compare_variable_patterns(self, vars1: List[str], vars2: List[str]) -> float:
        """Compare variable usage patterns - strict matching (no artificial boosts)"""
        set1 = set(vars1) if vars1 else set()
        set2 = set(vars2) if vars2 else set()
        
        if not set1 and not set2:
            return 1.0  # Both have no variables - perfect match
        
        # If one has variables and other doesn't, return low similarity
        if not set1 or not set2:
            return 0.2
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        # Standard Jaccard similarity - no artificial boosts
        return intersection / union if union > 0 else 0.0
    
    def _compare_function_signatures(self, sigs1: List[Dict], sigs2: List[Dict]) -> float:
        """Compare function signatures"""
        if not sigs1 and not sigs2:
            return 1.0
        if len(sigs1) != len(sigs2):
            return 0.0
        
        if not sigs1:
            return 1.0
        
        # Compare each signature
        similarities = []
        for s1, s2 in zip(sigs1, sigs2):
            sim = 0.0
            if s1['name'] == s2['name']:
                sim += 0.3
            if s1['num_args'] == s2['num_args']:
                sim += 0.3
            if s1['args'] == s2['args']:
                sim += 0.2
            if s1['has_defaults'] == s2['has_defaults']:
                sim += 0.1
            if s1['has_varargs'] == s2['has_varargs'] and s1['has_kwargs'] == s2['has_kwargs']:
                sim += 0.1
            similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _behavioral_similarity(self, code1: str, code2: str) -> float:
        """Calculate behavioral similarity"""
        try:
            code1_dedented = self.coerce_python_source_for_ast(code1)
            code2_dedented = self.coerce_python_source_for_ast(code2)
            
            # Extract function signatures and behavior patterns
            behavior1 = self._extract_behavior_patterns(code1_dedented)
            behavior2 = self._extract_behavior_patterns(code2_dedented)
            
            # Calculate similarity
            return self._compare_behavior_patterns(behavior1, behavior2)
            
        except Exception:
            # NO fallback - return 0.0 if behavioral analysis fails
            return 0.0
    
    def _extract_behavior_patterns(self, code: str) -> Dict[str, Any]:
        """Extract detailed behavior patterns from code"""
        try:
            tree = ast.parse(code)
            patterns = {
                'return_statements': [],
                'return_values': [],
                'exceptions': [],
                'exception_types': [],
                'conditionals': [],
                'conditional_conditions': [],
                'loops': [],
                'loop_types': [],
                'assignments': [],
                'assignment_targets': [],
                'function_calls': [],
                'call_targets': [],
                'side_effects': [],
                'modifications': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Return):
                    patterns['return_statements'].append(node.lineno if hasattr(node, 'lineno') else 0)
                    if node.value:
                        return_type = type(node.value).__name__
                        patterns['return_values'].append(return_type)
                        if isinstance(node.value, ast.Constant):
                            patterns['return_values'].append(f"constant_{type(node.value.value).__name__}")
                        elif isinstance(node.value, ast.Name):
                            patterns['return_values'].append(f"variable_{node.value.id}")
                
                elif isinstance(node, ast.Raise):
                    patterns['exceptions'].append(node.lineno if hasattr(node, 'lineno') else 0)
                    if node.exc:
                        if isinstance(node.exc, ast.Call):
                            if isinstance(node.exc.func, ast.Name):
                                patterns['exception_types'].append(node.exc.func.id)
                
                elif isinstance(node, ast.If):
                    patterns['conditionals'].append(node.lineno if hasattr(node, 'lineno') else 0)
                    if node.test:
                        test_str = ast.unparse(node.test) if hasattr(ast, 'unparse') else str(type(node.test).__name__)
                        patterns['conditional_conditions'].append(test_str[:50])
                
                elif isinstance(node, (ast.For, ast.While)):
                    patterns['loops'].append(node.lineno if hasattr(node, 'lineno') else 0)
                    patterns['loop_types'].append(type(node).__name__)
                
                elif isinstance(node, ast.Assign):
                    patterns['assignments'].append(node.lineno if hasattr(node, 'lineno') else 0)
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            patterns['assignment_targets'].append(target.id)
                        elif isinstance(target, ast.Attribute):
                            patterns['side_effects'].append(ast.unparse(target) if hasattr(ast, 'unparse') else str(target))
                            patterns['modifications'].append(target.attr if hasattr(target, 'attr') else '')
                
                elif isinstance(node, ast.Call):
                    patterns['function_calls'].append(node.lineno if hasattr(node, 'lineno') else 0)
                    if isinstance(node.func, ast.Name):
                        patterns['call_targets'].append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        patterns['call_targets'].append(node.func.attr)
            
            return patterns
            
        except Exception:
            return {}
    
    def _compare_behavior_patterns(self, patterns1: Dict[str, Any], patterns2: Dict[str, Any]) -> float:
        """Compare behavior patterns with strict matching"""
        if not patterns1 or not patterns2:
            if not patterns1 and not patterns2:
                return 0.5
            return 0.2
        
        similarities = []
        weights = {
            'return_statements': 0.15,
            'return_values': 0.15,
            'exceptions': 0.10,
            'exception_types': 0.10,
            'conditionals': 0.10,
            'conditional_conditions': 0.10,
            'loops': 0.10,
            'loop_types': 0.05,
            'assignments': 0.05,
            'assignment_targets': 0.05,
            'function_calls': 0.03,
            'call_targets': 0.02
        }
        
        for key in patterns1:
            if key not in patterns2:
                similarities.append((key, 0.0, weights.get(key, 0.05)))
                continue
            
            list1 = patterns1[key]
            list2 = patterns2[key]
            
            if not list1 and not list2:
                similarities.append((key, 1.0, weights.get(key, 0.05)))
            elif not list1 or not list2:
                similarities.append((key, 0.1, weights.get(key, 0.05)))
            else:
                set1 = set(list1)
                set2 = set(list2)
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                jaccard = intersection / union if union > 0 else 0.0
                
                if key in ['return_values', 'exception_types', 'call_targets', 'assignment_targets']:
                    if set1 == set2:
                        similarities.append((key, 1.0, weights.get(key, 0.05)))
                    else:
                        similarities.append((key, jaccard * 0.7, weights.get(key, 0.05)))
                else:
                    similarities.append((key, jaccard, weights.get(key, 0.05)))
        
        if not similarities:
            return 0.3
        
        total_weight = sum(weight for _, _, weight in similarities)
        weighted_sum = sum(sim * weight for _, sim, weight in similarities)
        final_similarity = weighted_sum / total_weight if total_weight > 0 else 0.3
        
        return min(final_similarity, 1.0)
    
