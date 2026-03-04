"""
Pattern-Based Specification Templates
Uses code pattern recognition to apply appropriate specification templates
"""

from typing import Dict, List, Any, Optional, Set
import ast
from collections import defaultdict
import re


class PatternTemplateMatcher:
    """Matches code patterns to appropriate specification templates"""
    
    def __init__(self):
        self.patterns = {
            'recursive_function': self._detect_recursive,
            'iterative_loop': self._detect_iterative,
            'stateful_method': self._detect_stateful,
            'pure_function': self._detect_pure,
            'exception_handler': self._detect_exception_handling,
            'data_transformer': self._detect_data_transformer,
            'validator': self._detect_validator,
            'factory': self._detect_factory,
            'accumulator': self._detect_accumulator,
            'search_function': self._detect_search
        }
        self.template_library = self._build_template_library()
    
    def match_pattern(self, code: str, function_name: str) -> List[str]:
        """Identify patterns in code"""
        try:
            tree = ast.parse(code)
            matched_patterns = []
            
            for pattern_name, detector in self.patterns.items():
                if detector(tree, function_name):
                    matched_patterns.append(pattern_name)
            
            return matched_patterns
        except Exception:
            return []
    
    def get_template_guidance(self, patterns: List[str]) -> str:
        """Get template-specific guidance for specification generation"""
        guidance_parts = []
        
        if 'recursive_function' in patterns:
            guidance_parts.append("""
PATTERN: Recursive Function
- Must document base case(s) explicitly
- Must document recursive case(s) with recursive call conditions
- Include termination conditions
- Specify stack behavior if relevant
""")
        
        if 'stateful_method' in patterns:
            guidance_parts.append("""
PATTERN: Stateful Method
- Must document all instance variable mutations
- Specify state transitions explicitly
- Include pre/post conditions for state
- Document side effects
""")
        
        if 'pure_function' in patterns:
            guidance_parts.append("""
PATTERN: Pure Function
- Document that function has no side effects
- Specify all inputs and outputs
- Document determinism
""")
        
        if 'exception_handler' in patterns:
            guidance_parts.append("""
PATTERN: Exception Handler
- Document all exception types that can be raised
- Specify conditions that trigger exceptions
- Include error handling paths
""")
        
        if 'data_transformer' in patterns:
            guidance_parts.append("""
PATTERN: Data Transformer
- Specify input/output data structures
- Document transformation rules
- Include edge cases for empty/invalid data
""")
        
        if 'validator' in patterns:
            guidance_parts.append("""
PATTERN: Validator Function
- Document all validation rules
- Specify what constitutes valid/invalid input
- Include return type (bool) and meaning
""")
        
        if 'accumulator' in patterns:
            guidance_parts.append("""
PATTERN: Accumulator Function
- Document initial value
- Specify accumulation operation
- Include termination condition
""")
        
        return "\n".join(guidance_parts)
    
    def _detect_recursive(self, tree: ast.AST, function_name: str) -> bool:
        """Detect recursive function calls"""
        func_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_names.add(node.name)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == function_name:
                    return True
                elif isinstance(node.func, ast.Attribute) and hasattr(node.func, 'attr'):
                    if node.func.attr == function_name:
                        return True
        return False
    
    def _detect_iterative(self, tree: ast.AST, function_name: str) -> bool:
        """Detect iterative loops"""
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                return True
        return False
    
    def _detect_stateful(self, tree: ast.AST, function_name: str) -> bool:
        """Detect state mutations"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        if isinstance(target.value, ast.Name) and target.value.id == 'self':
                            return True
        return False
    
    def _detect_pure(self, tree: ast.AST, function_name: str) -> bool:
        """Detect pure functions (no side effects)"""
        has_side_effects = False
        
        # Check for assignments to non-local variables
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Check if it's modifying a parameter or external state
                        # This is a simplified check
                        pass
                    elif isinstance(target, ast.Attribute):
                        has_side_effects = True
                        break
                if has_side_effects:
                    break
        
        # Check for I/O operations
        if not has_side_effects:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name in ['print', 'input', 'open', 'write', 'read']:
                            has_side_effects = True
                            break
        
        return not has_side_effects and isinstance(tree, ast.Module)
    
    def _detect_exception_handling(self, tree: ast.AST, function_name: str) -> bool:
        """Detect exception handling"""
        for node in ast.walk(tree):
            if isinstance(node, (ast.Raise, ast.Try)):
                return True
        return False
    
    def _detect_data_transformer(self, tree: ast.AST, function_name: str) -> bool:
        """Detect data transformation patterns"""
        # Look for patterns like map, filter, list comprehensions
        for node in ast.walk(tree):
            if isinstance(node, ast.ListComp) or isinstance(node, ast.DictComp):
                return True
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['map', 'filter', 'zip', 'enumerate']:
                        return True
        return False
    
    def _detect_validator(self, tree: ast.AST, function_name: str) -> bool:
        """Detect validation patterns"""
        # Look for patterns like starts_with, ends_with, contains, is_valid
        name_lower = function_name.lower()
        if any(keyword in name_lower for keyword in ['valid', 'check', 'is_', 'has_', 'verify']):
            return True
        
        # Check for boolean return
        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                if isinstance(node.value, (ast.NameConstant, ast.Constant)):
                    if isinstance(node.value.value, bool) if hasattr(node.value, 'value') else False:
                        return True
        return False
    
    def _detect_factory(self, tree: ast.AST, function_name: str) -> bool:
        """Detect factory patterns"""
        name_lower = function_name.lower()
        if 'factory' in name_lower or 'create' in name_lower or 'make' in name_lower:
            return True
        
        # Check if function creates and returns objects
        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                if isinstance(node.value, ast.Call):
                    return True
        return False
    
    def _detect_accumulator(self, tree: ast.AST, function_name: str) -> bool:
        """Detect accumulator patterns"""
        # Look for +=, -=, *=, /=
        for node in ast.walk(tree):
            if isinstance(node, ast.AugAssign):
                return True
        return False
    
    def _detect_search(self, tree: ast.AST, function_name: str) -> bool:
        """Detect search patterns"""
        name_lower = function_name.lower()
        if any(keyword in name_lower for keyword in ['find', 'search', 'get', 'lookup', 'contains']):
            return True
        return False
    
    def _build_template_library(self) -> Dict[str, Dict[str, Any]]:
        """Build library of specification templates"""
        return {
            'recursive_function': {
                'required_fields': ['base_case', 'recursive_case', 'termination'],
                'focus_areas': ['recursion depth', 'stack behavior', 'base conditions']
            },
            'stateful_method': {
                'required_fields': ['state_mutations', 'pre_conditions', 'post_conditions'],
                'focus_areas': ['instance variables', 'side effects', 'state transitions']
            },
            'pure_function': {
                'required_fields': ['inputs', 'outputs', 'determinism'],
                'focus_areas': ['no side effects', 'referential transparency']
            },
            'data_transformer': {
                'required_fields': ['input_structure', 'output_structure', 'transformation_rules'],
                'focus_areas': ['data formats', 'edge cases', 'empty data handling']
            }
        }

