"""
Advanced Analysis Agents
Specialized agents for different types of code analysis
"""

import ast
import json
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
                    dependencies['imports'].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies['imports'].append(node.module)
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


class SemanticSimilarityAnalyzer:
    """Advanced semantic similarity analysis"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=2000,
            stop_words=None
        )
    
    def calculate_semantic_similarity(self, code1: str, code2: str) -> Dict[str, float]:
        """Calculate comprehensive semantic similarity"""
        return {
            'textual_similarity': self._textual_similarity(code1, code2),
            'structural_similarity': self._structural_similarity(code1, code2),
            'behavioral_similarity': self._behavioral_similarity(code1, code2),
            'semantic_similarity': self._semantic_similarity(code1, code2),
            'overall_similarity': 0.0  # Will be calculated
        }
    
    def _textual_similarity(self, code1: str, code2: str) -> float:
        """Calculate textual similarity"""
        import difflib
        return difflib.SequenceMatcher(None, code1, code2).ratio()
    
    def _structural_similarity(self, code1: str, code2: str) -> float:
        """Calculate structural similarity using AST"""
        try:
            tree1 = ast.parse(code1)
            tree2 = ast.parse(code2)
            
            # Convert ASTs to structural representations
            struct1 = self._ast_to_structure(tree1)
            struct2 = self._ast_to_structure(tree2)
            
            # Calculate similarity
            import difflib
            return difflib.SequenceMatcher(None, struct1, struct2).ratio()
            
        except SyntaxError:
            return 0.0
    
    def _behavioral_similarity(self, code1: str, code2: str) -> float:
        """Calculate behavioral similarity"""
        try:
            # Extract function signatures and behavior patterns
            behavior1 = self._extract_behavior_patterns(code1)
            behavior2 = self._extract_behavior_patterns(code2)
            
            # Calculate similarity
            return self._compare_behavior_patterns(behavior1, behavior2)
            
        except Exception:
            return 0.0
    
    def _semantic_similarity(self, code1: str, code2: str) -> float:
        """Calculate semantic similarity using TF-IDF"""
        try:
            # Preprocess code for semantic analysis
            processed1 = self._preprocess_for_semantic(code1)
            processed2 = self._preprocess_for_semantic(code2)
            
            # Create TF-IDF vectors
            corpus = [processed1, processed2]
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
            
        except Exception:
            return 0.0
    
    def _ast_to_structure(self, tree: ast.AST) -> str:
        """Convert AST to structural representation"""
        structure = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                structure.append(f"FUNC:{node.name}")
            elif isinstance(node, ast.ClassDef):
                structure.append(f"CLASS:{node.name}")
            elif isinstance(node, ast.If):
                structure.append("IF")
            elif isinstance(node, ast.While):
                structure.append("WHILE")
            elif isinstance(node, ast.For):
                structure.append("FOR")
            elif isinstance(node, ast.Return):
                structure.append("RETURN")
        
        return " ".join(structure)
    
    def _extract_behavior_patterns(self, code: str) -> Dict[str, Any]:
        """Extract behavior patterns from code"""
        try:
            tree = ast.parse(code)
            patterns = {
                'input_processing': [],
                'output_generation': [],
                'error_handling': [],
                'control_flow': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Analyze function behavior
                    for child in node.body:
                        if isinstance(child, ast.Return):
                            patterns['output_generation'].append('return')
                        elif isinstance(child, ast.Raise):
                            patterns['error_handling'].append('raise')
                        elif isinstance(child, ast.If):
                            patterns['control_flow'].append('conditional')
            
            return patterns
            
        except Exception:
            return {}
    
    def _compare_behavior_patterns(self, patterns1: Dict[str, Any], patterns2: Dict[str, Any]) -> float:
        """Compare behavior patterns"""
        if not patterns1 or not patterns2:
            return 0.0
        
        similarities = []
        for key in patterns1:
            if key in patterns2:
                list1 = patterns1[key]
                list2 = patterns2[key]
                if list1 and list2:
                    intersection = len(set(list1) & set(list2))
                    union = len(set(list1) | set(list2))
                    similarities.append(intersection / union if union > 0 else 0.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _preprocess_for_semantic(self, code: str) -> str:
        """Preprocess code for semantic analysis"""
        # Remove comments and docstrings
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)
        
        # Extract meaningful tokens
        tokens = []
        for line in code.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                tokens.extend(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', line))
        
        return ' '.join(tokens)
