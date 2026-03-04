"""
Abstract Interpretation for Specification Generation
Infers program properties and invariants without executing all paths
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import ast
from enum import Enum


class AbstractValue(Enum):
    """Abstract values for abstract interpretation"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    ZERO = "zero"
    NON_NEGATIVE = "non_negative"
    NON_POSITIVE = "non_positive"
    EMPTY = "empty"
    NON_EMPTY = "non_empty"
    NONE = "none"
    NOT_NONE = "not_none"
    TOP = "top"  # Unknown/any value
    BOTTOM = "bottom"  # Impossible value


class AbstractInterpreter:
    """Performs abstract interpretation to infer program properties"""
    
    def infer_invariants(self, code: str) -> Dict[str, Any]:
        """Infer loop invariants, value ranges, and conditions"""
        try:
            tree = ast.parse(code)
            invariants = {
                'loop_invariants': [],
                'value_ranges': {},
                'conditions': [],
                'preconditions': [],
                'postconditions': []
            }
            
            # Analyze loops
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    loop_inv = self._analyze_loop(node, tree)
                    if loop_inv:
                        invariants['loop_invariants'].append(loop_inv)
                elif isinstance(node, ast.While):
                    loop_inv = self._analyze_while_loop(node, tree)
                    if loop_inv:
                        invariants['loop_invariants'].append(loop_inv)
            
            # Analyze value ranges
            value_ranges = self._infer_value_ranges(tree)
            invariants['value_ranges'] = value_ranges
            
            # Analyze conditions
            conditions = self._extract_conditions(tree)
            invariants['conditions'] = conditions
            
            return invariants
        
        except Exception:
            return {
                'loop_invariants': [],
                'value_ranges': {},
                'conditions': [],
                'preconditions': [],
                'postconditions': []
            }
    
    def _analyze_loop(self, loop_node: ast.For, tree: ast.AST) -> Optional[Dict[str, Any]]:
        """Analyze a for loop to infer invariants"""
        # Extract loop variable
        if isinstance(loop_node.target, ast.Name):
            var_name = loop_node.target.id
        else:
            return None
        
        # Check what the loop iterates over
        iter_expr = loop_node.iter
        invariant = {
            'variable': var_name,
            'type': 'for_loop',
            'invariant': f"{var_name} iterates over collection"
        }
        
        # Check for common patterns
        if isinstance(iter_expr, ast.Call):
            if isinstance(iter_expr.func, ast.Name):
                if iter_expr.func.id == 'range':
                    invariant['invariant'] = f"{var_name} is in range"
                    invariant['range_based'] = True
        
        # Check loop body for patterns
        has_accumulation = any(
            isinstance(stmt, ast.AugAssign) for stmt in ast.walk(loop_node)
        )
        if has_accumulation:
            invariant['has_accumulation'] = True
            invariant['invariant'] += " with accumulation"
        
        return invariant
    
    def _analyze_while_loop(self, loop_node: ast.While, tree: ast.AST) -> Optional[Dict[str, Any]]:
        """Analyze a while loop to infer invariants"""
        # Extract condition
        condition_str = ast.unparse(loop_node.test) if hasattr(ast, 'unparse') else str(loop_node.test)
        
        return {
            'type': 'while_loop',
            'condition': condition_str,
            'invariant': f"Loop continues while: {condition_str}"
        }
    
    def _infer_value_ranges(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Infer value ranges for variables"""
        ranges = {}
        
        # Look for comparisons that indicate ranges
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                # Check if comparing with zero
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Constant):
                        if comparator.value == 0:
                            if isinstance(node.left, ast.Name):
                                var = node.left.id
                                # Check operator
                                for op in node.ops:
                                    if isinstance(op, ast.Gt):
                                        ranges[var] = ranges.get(var, []) + ['positive']
                                    elif isinstance(op, ast.Lt):
                                        ranges[var] = ranges.get(var, []) + ['negative']
                                    elif isinstance(op, ast.GtE):
                                        ranges[var] = ranges.get(var, []) + ['non_negative']
                                    elif isinstance(op, ast.LtE):
                                        ranges[var] = ranges.get(var, []) + ['non_positive']
        
        return ranges
    
    def _extract_conditions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract conditions that determine function behavior"""
        conditions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                condition_str = ast.unparse(node.test) if hasattr(ast, 'unparse') else str(node.test)
                conditions.append({
                    'type': 'conditional',
                    'condition': condition_str,
                    'has_else': len(node.orelse) > 0
                })
        
        return conditions
    
    def generate_invariant_specification(self, invariants: Dict[str, Any]) -> str:
        """Generate specification text from inferred invariants"""
        spec_parts = []
        
        if invariants.get('loop_invariants'):
            spec_parts.append("Loop Invariants:")
            for inv in invariants['loop_invariants']:
                spec_parts.append(f"  - {inv.get('invariant', 'Unknown invariant')}")
        
        if invariants.get('value_ranges'):
            spec_parts.append("Value Ranges:")
            for var, ranges in invariants['value_ranges'].items():
                spec_parts.append(f"  - {var}: {', '.join(ranges)}")
        
        if invariants.get('conditions'):
            spec_parts.append("Key Conditions:")
            for cond in invariants['conditions'][:5]:  # Limit to 5
                spec_parts.append(f"  - {cond.get('condition', 'Unknown')}")
        
        return "\n".join(spec_parts) if spec_parts else ""

