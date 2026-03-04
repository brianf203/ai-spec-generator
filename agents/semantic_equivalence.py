"""
Semantic Equivalence Detection
Detects when code is semantically equivalent but structurally different
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import ast
import re
from collections import defaultdict


class SemanticEquivalenceDetector:
    """Detects semantic equivalence between original and regenerated code"""
    
    def __init__(self):
        self.equivalence_patterns = {
            'list_comprehension_vs_loop': self._compare_list_comp_to_loop,
            'ternary_vs_if_else': self._compare_ternary_to_if_else,
            'lambda_vs_function': self._compare_lambda_to_function,
            'set_vs_list_uniqueness': self._compare_set_to_list,
            'different_variable_names': self._compare_variable_names_semantically
        }
    
    def detect_semantic_equivalence(
        self,
        original_code: str,
        regenerated_code: str
    ) -> Tuple[bool, float, List[str]]:
        """
        Detect if code is semantically equivalent
        Returns: (is_equivalent, confidence, reasons)
        """
        try:
            # Dedent code to handle indented code from class methods
            from textwrap import dedent
            orig_clean = dedent(original_code).strip() or original_code.strip()
            regen_clean = dedent(regenerated_code).strip() or regenerated_code.strip()
            
            orig_tree = ast.parse(orig_clean)
            regen_tree = ast.parse(regen_clean)
        except SyntaxError:
            return False, 0.0, ["One or both code snippets have syntax errors"]
        
        # Check basic structural equivalence first
        structural_score = self._structural_equivalence_score(orig_tree, regen_tree)
        
        if structural_score > 0.95:
            return True, structural_score, ["High structural similarity"]
        
        # Check semantic equivalence patterns
        equivalence_checks = []
        confidence_scores = []
        
        for pattern_name, checker in self.equivalence_patterns.items():
            is_equiv, confidence, reason = checker(orig_tree, regen_tree)
            if is_equiv:
                equivalence_checks.append(reason)
                confidence_scores.append(confidence)
        
        if equivalence_checks:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            return True, avg_confidence, equivalence_checks
        
        # Check if outputs would be the same (heuristic)
        output_equivalence = self._check_output_equivalence(orig_tree, regen_tree)
        if output_equivalence[0]:
            return True, output_equivalence[1], ["Output equivalence detected"]
        
        return False, structural_score, []
    
    def _structural_equivalence_score(
        self,
        tree1: ast.AST,
        tree2: ast.AST
    ) -> float:
        """Calculate basic structural equivalence score"""
        # Count node types
        nodes1 = defaultdict(int)
        nodes2 = defaultdict(int)
        
        for node in ast.walk(tree1):
            nodes1[type(node).__name__] += 1
        
        for node in ast.walk(tree2):
            nodes2[type(node).__name__] += 1
        
        # Calculate similarity
        all_types = set(nodes1.keys()) | set(nodes2.keys())
        if not all_types:
            return 1.0
        
        similarities = []
        for node_type in all_types:
            count1 = nodes1.get(node_type, 0)
            count2 = nodes2.get(node_type, 0)
            if count1 == 0 and count2 == 0:
                continue
            similarity = min(count1, count2) / max(count1, count2) if max(count1, count2) > 0 else 1.0
            similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _compare_list_comp_to_loop(
        self,
        tree1: ast.AST,
        tree2: ast.AST
    ) -> Tuple[bool, float, str]:
        """Check if list comprehension and loop are equivalent"""
        has_list_comp1 = any(isinstance(n, ast.ListComp) for n in ast.walk(tree1))
        has_list_comp2 = any(isinstance(n, ast.ListComp) for n in ast.walk(tree2))
        has_for_loop1 = any(isinstance(n, ast.For) for n in ast.walk(tree1))
        has_for_loop2 = any(isinstance(n, ast.For) for n in ast.walk(tree2))
        
        if (has_list_comp1 and has_for_loop2) or (has_list_comp2 and has_for_loop1):
            # Could be semantically equivalent - would need deeper analysis
            return True, 0.8, "List comprehension vs loop pattern detected"
        
        return False, 0.0, ""
    
    def _compare_ternary_to_if_else(
        self,
        tree1: ast.AST,
        tree2: ast.AST
    ) -> Tuple[bool, float, str]:
        """Check if ternary and if-else are equivalent"""
        has_ternary1 = any(
            isinstance(n, ast.IfExp) for n in ast.walk(tree1)
        )
        has_if1 = any(
            isinstance(n, ast.If) and n.orelse for n in ast.walk(tree1)
        )
        has_ternary2 = any(
            isinstance(n, ast.IfExp) for n in ast.walk(tree2)
        )
        has_if2 = any(
            isinstance(n, ast.If) and n.orelse for n in ast.walk(tree2)
        )
        
        if (has_ternary1 and has_if2) or (has_ternary2 and has_if1):
            return True, 0.85, "Ternary vs if-else pattern detected"
        
        return False, 0.0, ""
    
    def _compare_lambda_to_function(
        self,
        tree1: ast.AST,
        tree2: ast.AST
    ) -> Tuple[bool, float, str]:
        """Check if lambda and named function are equivalent"""
        has_lambda1 = any(isinstance(n, ast.Lambda) for n in ast.walk(tree1))
        has_lambda2 = any(isinstance(n, ast.Lambda) for n in ast.walk(tree2))
        has_function1 = any(isinstance(n, ast.FunctionDef) for n in ast.walk(tree1) if isinstance(n, ast.FunctionDef) and n.name != 'self')
        has_function2 = any(isinstance(n, ast.FunctionDef) for n in ast.walk(tree2) if isinstance(n, ast.FunctionDef) and n.name != 'self')
        
        # This is a heuristic - would need deeper analysis
        if (has_lambda1 and has_function2) or (has_lambda2 and has_function1):
            return True, 0.7, "Lambda vs function pattern detected"
        
        return False, 0.0, ""
    
    def _compare_set_to_list(
        self,
        tree1: ast.AST,
        tree2: ast.AST
    ) -> Tuple[bool, float, str]:
        """Check if set() and list uniqueness operations are equivalent"""
        # Look for set() calls vs list(set(...)) patterns
        has_set1 = any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'set'
            for n in ast.walk(tree1)
        )
        has_set2 = any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'set'
            for n in ast.walk(tree2)
        )
        
        # Simplified check
        if has_set1 != has_set2:
            return True, 0.75, "Set vs list uniqueness pattern detected"
        
        return False, 0.0, ""
    
    def _compare_variable_names_semantically(
        self,
        tree1: ast.AST,
        tree2: ast.AST
    ) -> Tuple[bool, float, str]:
        """Check if variable names are semantically similar but different"""
        # Extract variable names
        vars1 = set()
        vars2 = set()
        
        for node in ast.walk(tree1):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                vars1.add(node.id)
        
        for node in ast.walk(tree2):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                vars2.add(node.id)
        
        # Check for semantic similarity (same length, similar patterns)
        # This is a very simplified check
        if vars1 and vars2 and len(vars1) == len(vars2):
            # Could be semantically equivalent with different names
            return True, 0.6, "Variable names differ but structure similar"
        
        return False, 0.0, ""
    
    def _check_output_equivalence(
        self,
        tree1: ast.AST,
        tree2: ast.AST
    ) -> Tuple[bool, float]:
        """Heuristic check for output equivalence"""
        # Check return statement patterns
        returns1 = [n for n in ast.walk(tree1) if isinstance(n, ast.Return)]
        returns2 = [n for n in ast.walk(tree2) if isinstance(n, ast.Return)]
        
        if len(returns1) != len(returns2):
            return False, 0.0
        
        # Check if return types are similar
        return_types1 = [type(r.value).__name__ if r.value else 'None' for r in returns1]
        return_types2 = [type(r.value).__name__ if r.value else 'None' for r in returns2]
        
        if return_types1 == return_types2:
            return True, 0.8
        
        return False, 0.0
    
    def adjust_similarity_for_equivalence(
        self,
        original_code: str,
        regenerated_code: str,
        structural_similarity: float,
        behavioral_similarity: float
    ) -> Tuple[float, float]:
        """Adjust similarity scores if code is semantically equivalent"""
        is_equiv, confidence, reasons = self.detect_semantic_equivalence(
            original_code,
            regenerated_code
        )
        
        if is_equiv and confidence > 0.8:
            # Boost similarity scores based on confidence (conservative)
            # Only apply if confidence is high (>0.8) and cap boosts lower
            structural_boost = min(0.05, confidence * 0.1)  # Max 5% boost
            behavioral_boost = min(0.08, confidence * 0.15)  # Max 8% boost
            
            adjusted_structural = min(1.0, structural_similarity + structural_boost)
            adjusted_behavioral = min(1.0, behavioral_similarity + behavioral_boost)
            
            return adjusted_structural, adjusted_behavioral
        
        return structural_similarity, behavioral_similarity

