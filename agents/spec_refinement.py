"""
Advanced Specification Refinement System
Implements iterative refinement, pattern matching, and constraint propagation
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import ast
import re
import json
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class RefinementTarget:
    """Target area for specification refinement"""
    type: str  # 'structural', 'behavioral', 'variable_names', 'control_flow', 'edge_cases'
    priority: float
    current_gap: float
    specific_issues: List[str]
    suggested_fixes: List[str]


class SpecificationRefinementEngine:
    """Advanced engine for iteratively refining specifications based on similarity gaps"""
    
    def __init__(self):
        self.refinement_history: Dict[str, List[Dict[str, Any]]] = {}
        self.pattern_library: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def analyze_refinement_opportunities(
        self,
        specification: Dict[str, Any],
        original_code: str,
        regenerated_code: str,
        similarity_metrics: Dict[str, float],
        test_results: Optional[Dict[str, Any]] = None
    ) -> List[RefinementTarget]:
        """Identify specific areas where specification needs refinement"""
        targets = []
        
        primary_sim = similarity_metrics.get('primary_similarity', 1.0)
        structural_sim = similarity_metrics.get('structural_similarity', 0.0)
        behavioral_sim = similarity_metrics.get('behavioral_similarity', 0.0)
        behavioral_test_sim = similarity_metrics.get('behavioral_test_similarity', 0.0)
        
        # Structural refinement targets
        if structural_sim < 0.85:
            structural_issues = self._analyze_structural_gaps(original_code, regenerated_code, specification)
            targets.append(RefinementTarget(
                type='structural',
                priority=1.0 - structural_sim,
                current_gap=1.0 - structural_sim,
                specific_issues=structural_issues,
                suggested_fixes=self._suggest_structural_fixes(structural_issues)
            ))
        
        # Behavioral refinement targets
        if behavioral_sim < 0.85:
            behavioral_issues = self._analyze_behavioral_gaps(original_code, regenerated_code, specification)
            targets.append(RefinementTarget(
                type='behavioral',
                priority=1.0 - behavioral_sim,
                current_gap=1.0 - behavioral_sim,
                specific_issues=behavioral_issues,
                suggested_fixes=self._suggest_behavioral_fixes(behavioral_issues)
            ))
        
        # Variable name refinement
        var_name_issues = self._analyze_variable_name_gaps(original_code, regenerated_code, specification)
        if var_name_issues:
            targets.append(RefinementTarget(
                type='variable_names',
                priority=0.8,
                current_gap=len(var_name_issues) / 10.0,
                specific_issues=var_name_issues,
                suggested_fixes=self._suggest_variable_name_fixes(var_name_issues)
            ))
        
        # Control flow refinement
        control_flow_issues = self._analyze_control_flow_gaps(original_code, regenerated_code, specification)
        if control_flow_issues:
            targets.append(RefinementTarget(
                type='control_flow',
                priority=0.7,
                current_gap=len(control_flow_issues) / 5.0,
                specific_issues=control_flow_issues,
                suggested_fixes=self._suggest_control_flow_fixes(control_flow_issues)
            ))
        
        # Edge case refinement based on test results
        if test_results:
            edge_case_issues = self._analyze_edge_case_gaps(test_results, specification)
            if edge_case_issues:
                targets.append(RefinementTarget(
                    type='edge_cases',
                    priority=0.6,
                    current_gap=len(edge_case_issues) / 8.0,
                    specific_issues=edge_case_issues,
                    suggested_fixes=self._suggest_edge_case_fixes(edge_case_issues)
                ))
        
        # CRITICAL FIX: If primary_similarity < 0.85 but no specific targets were found,
        # create a general refinement target to ensure refinement happens
        if primary_sim < 0.85 and not targets:
            # Analyze what might be wrong - check all aspects
            all_issues = []
            structural_issues = self._analyze_structural_gaps(original_code, regenerated_code, specification)
            behavioral_issues = self._analyze_behavioral_gaps(original_code, regenerated_code, specification)
            var_name_issues = self._analyze_variable_name_gaps(original_code, regenerated_code, specification)
            control_flow_issues = self._analyze_control_flow_gaps(original_code, regenerated_code, specification)
            
            if structural_issues:
                all_issues.extend(structural_issues)
            if behavioral_issues:
                all_issues.extend(behavioral_issues)
            if var_name_issues:
                all_issues.extend(var_name_issues)
            if control_flow_issues:
                all_issues.extend(control_flow_issues)
            
            # Create a general refinement target
            targets.append(RefinementTarget(
                type='general',
                priority=1.0 - primary_sim,
                current_gap=1.0 - primary_sim,
                specific_issues=all_issues[:10] if all_issues else [f"Primary similarity is {primary_sim:.1%}, needs improvement"],
                suggested_fixes=self._suggest_general_fixes(primary_sim, all_issues)
            ))
        
        # Sort by priority
        targets.sort(key=lambda x: x.priority, reverse=True)
        return targets
    
    def refine_specification(
        self,
        specification: Dict[str, Any],
        targets: List[RefinementTarget],
        original_code: str
    ) -> Dict[str, Any]:
        """Apply refinements to specification based on targets"""
        refined_spec = specification.copy()
        
        for target in targets[:5]:  # Focus on top 5 targets
            if target.type == 'structural':
                refined_spec = self._refine_structural(refined_spec, target, original_code)
            elif target.type == 'variable_names':
                refined_spec = self._refine_variable_names(refined_spec, target, original_code)
            elif target.type == 'control_flow':
                refined_spec = self._refine_control_flow(refined_spec, target, original_code)
            elif target.type == 'edge_cases':
                refined_spec = self._refine_edge_cases(refined_spec, target)
            elif target.type == 'general':
                # For general refinement, apply all available refinements
                refined_spec = self._refine_general(refined_spec, target, original_code)
        
        # Add refinement metadata
        refined_spec['refinement_history'] = refined_spec.get('refinement_history', []) + [
            {
                'targets_addressed': [t.type for t in targets[:5]],
                'priorities': [t.priority for t in targets[:5]]
            }
        ]
        
        return refined_spec
    
    def _analyze_structural_gaps(
        self,
        original_code: str,
        regenerated_code: str,
        specification: Dict[str, Any]
    ) -> List[str]:
        """Analyze structural differences between original and regenerated code"""
        issues = []
        
        try:
            orig_tree = ast.parse(original_code)
            regen_tree = ast.parse(regenerated_code) if regenerated_code else None
            
            if not regen_tree:
                return ["Regenerated code is invalid or empty"]
            
            # Compare function signatures
            orig_funcs = [n for n in ast.walk(orig_tree) if isinstance(n, ast.FunctionDef)]
            regen_funcs = [n for n in ast.walk(regen_tree) if isinstance(n, ast.FunctionDef)]
            
            if len(orig_funcs) != len(regen_funcs):
                issues.append(f"Function count mismatch: original has {len(orig_funcs)}, regenerated has {len(regen_funcs)}")
            
            if orig_funcs and regen_funcs:
                orig_func = orig_funcs[0]
                regen_func = regen_funcs[0]
                
                # Check parameter names
                orig_params = [arg.arg for arg in orig_func.args.args]
                regen_params = [arg.arg for arg in regen_func.args.args]
                if orig_params != regen_params:
                    issues.append(f"Parameter names differ: original {orig_params} vs regenerated {regen_params}")
                
                # Check return statements
                orig_returns = [n for n in ast.walk(orig_func) if isinstance(n, ast.Return)]
                regen_returns = [n for n in ast.walk(regen_func) if isinstance(n, ast.Return)]
                if len(orig_returns) != len(regen_returns):
                    issues.append(f"Return statement count differs: {len(orig_returns)} vs {len(regen_returns)}")
            
            # Check AST node type distribution
            orig_node_types = defaultdict(int)
            regen_node_types = defaultdict(int)
            
            for node in ast.walk(orig_tree):
                orig_node_types[type(node).__name__] += 1
            for node in ast.walk(regen_tree):
                regen_node_types[type(node).__name__] += 1
            
            for node_type, orig_count in orig_node_types.items():
                regen_count = regen_node_types.get(node_type, 0)
                if abs(orig_count - regen_count) > 2:
                    issues.append(f"Node type {node_type} count differs significantly: {orig_count} vs {regen_count}")
        
        except SyntaxError as e:
            issues.append(f"Syntax error in code: {str(e)}")
        except Exception as e:
            issues.append(f"Error analyzing structural gaps: {str(e)}")
        
        return issues
    
    def _analyze_behavioral_gaps(
        self,
        original_code: str,
        regenerated_code: str,
        specification: Dict[str, Any]
    ) -> List[str]:
        """Analyze behavioral differences"""
        issues = []
        
        # Check if side effects are documented
        spec_side_effects = specification.get('side_effects', {})
        
        # Check for state mutations in original code
        try:
            orig_tree = ast.parse(original_code)
            has_mutations = any(
                isinstance(node, ast.Assign) and
                isinstance(node.targets[0], ast.Attribute) if node.targets else False
                for node in ast.walk(orig_tree)
            )
            
            if has_mutations and not spec_side_effects:
                issues.append("Code has state mutations but spec doesn't document side effects")
        except Exception:
            pass
        
        return issues
    
    def _analyze_variable_name_gaps(
        self,
        original_code: str,
        regenerated_code: str,
        specification: Dict[str, Any]
    ) -> List[str]:
        """Identify variable name mismatches"""
        issues = []
        
        try:
            orig_tree = ast.parse(original_code)
            regen_tree = ast.parse(regenerated_code) if regenerated_code else None
            
            if not regen_tree:
                return issues
            
            # Extract variable names from both
            orig_vars = set()
            regen_vars = set()
            
            for node in ast.walk(orig_tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    orig_vars.add(node.id)
            
            for node in ast.walk(regen_tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    regen_vars.add(node.id)
            
            # Find mismatches
            missing_in_regen = orig_vars - regen_vars
            extra_in_regen = regen_vars - orig_vars
            
            if missing_in_regen:
                issues.append(f"Variables missing in regenerated code: {list(missing_in_regen)[:5]}")
            if extra_in_regen:
                issues.append(f"Extra variables in regenerated code: {list(extra_in_regen)[:5]}")
            
        except Exception:
            pass
        
        return issues
    
    def _analyze_control_flow_gaps(
        self,
        original_code: str,
        regenerated_code: str,
        specification: Dict[str, Any]
    ) -> List[str]:
        """Analyze control flow structure differences"""
        issues = []
        
        try:
            orig_tree = ast.parse(original_code)
            regen_tree = ast.parse(regenerated_code) if regenerated_code else None
            
            if not regen_tree:
                return issues
            
            # Count control flow constructs
            orig_control = {
                'if': len([n for n in ast.walk(orig_tree) if isinstance(n, ast.If)]),
                'for': len([n for n in ast.walk(orig_tree) if isinstance(n, ast.For)]),
                'while': len([n for n in ast.walk(orig_tree) if isinstance(n, ast.While)]),
                'try': len([n for n in ast.walk(orig_tree) if isinstance(n, ast.Try)]),
            }
            
            regen_control = {
                'if': len([n for n in ast.walk(regen_tree) if isinstance(n, ast.If)]),
                'for': len([n for n in ast.walk(regen_tree) if isinstance(n, ast.For)]),
                'while': len([n for n in ast.walk(regen_tree) if isinstance(n, ast.While)]),
                'try': len([n for n in ast.walk(regen_tree) if isinstance(n, ast.Try)]),
            }
            
            for construct, orig_count in orig_control.items():
                regen_count = regen_control.get(construct, 0)
                if orig_count != regen_count:
                    issues.append(f"{construct} statement count differs: {orig_count} vs {regen_count}")
        
        except Exception:
            pass
        
        return issues
    
    def _analyze_edge_case_gaps(
        self,
        test_results: Dict[str, Any],
        specification: Dict[str, Any]
    ) -> List[str]:
        """Analyze missing edge cases based on test failures"""
        issues = []
        
        failures = test_results.get('failures', [])
        missing_lines = test_results.get('missing_lines', [])
        missing_branches = test_results.get('missing_branches', [])
        
        if missing_lines:
            issues.append(f"Lines not covered by tests: {missing_lines[:5]}")
        
        if missing_branches:
            issues.append(f"Branches not covered: {len(missing_branches)} branch(es)")
        
        if failures:
            # Analyze failure patterns
            exception_failures = [f for f in failures if f.get('regenerated_exception') or f.get('original_exception')]
            if exception_failures:
                issues.append(f"{len(exception_failures)} test(s) fail due to exception handling differences")
        
        return issues
    
    def _suggest_structural_fixes(self, issues: List[str]) -> List[str]:
        """Suggest fixes for structural issues"""
        fixes = []
        for issue in issues:
            if 'parameter names' in issue.lower():
                fixes.append("Explicitly specify exact parameter names in specification")
            elif 'return statement' in issue.lower():
                fixes.append("Document all return statements and their conditions")
            elif 'function count' in issue.lower():
                fixes.append("Ensure specification clearly defines function structure")
        return fixes
    
    def _suggest_behavioral_fixes(self, issues: List[str]) -> List[str]:
        """Suggest fixes for behavioral issues"""
        fixes = []
        for issue in issues:
            if 'side effects' in issue.lower():
                fixes.append("Document all state mutations and side effects explicitly")
        return fixes
    
    def _suggest_variable_name_fixes(self, issues: List[str]) -> List[str]:
        """Suggest fixes for variable name issues"""
        return [
            "Extract and explicitly list all variable names from original code",
            "Include variable name mapping in specification",
            "Specify variable names as mandatory, not suggestions"
        ]
    
    def _suggest_control_flow_fixes(self, issues: List[str]) -> List[str]:
        """Suggest fixes for control flow issues"""
        return [
            "Document exact control flow structure (if/else, loops, try/except)",
            "Specify order and nesting of control structures",
            "Include conditions for each branch"
        ]
    
    def _suggest_edge_case_fixes(self, issues: List[str]) -> List[str]:
        """Suggest fixes for edge case issues"""
        return [
            "Add edge cases to test matrix for uncovered lines",
            "Document boundary conditions explicitly",
            "Include exception handling scenarios"
        ]
    
    def _suggest_general_fixes(self, primary_sim: float, issues: List[str]) -> List[str]:
        """Suggest general fixes when primary similarity is low but no specific issues identified"""
        fixes = [
            "Enhance specification with more detailed variable name requirements",
            "Add explicit control flow structure documentation",
            "Include more comprehensive test case examples",
            "Strengthen structural similarity requirements",
            "Add behavioral equivalence constraints"
        ]
        if issues:
            fixes.extend([
                f"Address identified issues: {', '.join(issues[:3])}",
                "Review and refine all specification sections"
            ])
        return fixes
    
    def _refine_general(
        self,
        specification: Dict[str, Any],
        target: RefinementTarget,
        original_code: str
    ) -> Dict[str, Any]:
        """Apply general refinements when no specific type is identified"""
        spec = specification.copy()
        
        # Apply structural refinements
        structural_target = RefinementTarget(
            type='structural',
            priority=target.priority,
            current_gap=target.current_gap,
            specific_issues=target.specific_issues,
            suggested_fixes=target.suggested_fixes
        )
        spec = self._refine_structural(spec, structural_target, original_code)
        
        # Apply variable name refinements
        var_name_issues = self._analyze_variable_name_gaps(original_code, "", spec)
        if var_name_issues:
            var_target = RefinementTarget(
                type='variable_names',
                priority=0.8,
                current_gap=len(var_name_issues) / 10.0,
                specific_issues=var_name_issues,
                suggested_fixes=self._suggest_variable_name_fixes(var_name_issues)
            )
            spec = self._refine_variable_names(spec, var_target, original_code)
        
        # Apply control flow refinements
        control_flow_issues = self._analyze_control_flow_gaps(original_code, "", spec)
        if control_flow_issues:
            cf_target = RefinementTarget(
                type='control_flow',
                priority=0.7,
                current_gap=len(control_flow_issues) / 5.0,
                specific_issues=control_flow_issues,
                suggested_fixes=self._suggest_control_flow_fixes(control_flow_issues)
            )
            spec = self._refine_control_flow(spec, cf_target, original_code)
        
        # Add a note about general refinement
        if 'refinement_notes' not in spec:
            spec['refinement_notes'] = []
        spec['refinement_notes'].append(f"General refinement applied due to primary similarity {target.current_gap:.1%} below threshold")
        
        return spec
    
    def _refine_structural(
        self,
        specification: Dict[str, Any],
        target: RefinementTarget,
        original_code: str
    ) -> Dict[str, Any]:
        """Refine structural aspects of specification"""
        spec = specification.copy()
        
        # Extract exact function signature from original code
        try:
            tree = ast.parse(original_code)
            funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            if funcs:
                func = funcs[0]
                params = [arg.arg for arg in func.args.args]
                
                # Update specification with exact parameter names
                if 'signature' not in spec:
                    spec['signature'] = {}
                spec['signature']['parameters'] = [
                    {'name': p, 'exact_match_required': True} for p in params
                ]
        except Exception:
            pass
        
        return spec
    
    def _refine_variable_names(
        self,
        specification: Dict[str, Any],
        target: RefinementTarget,
        original_code: str
    ) -> Dict[str, Any]:
        """Refine variable names in specification"""
        spec = specification.copy()
        
        # Extract all variable names from original code
        try:
            tree = ast.parse(original_code)
            var_names = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_names.append({
                                'name': target.id,
                                'exact_match_required': True,
                                'line': getattr(node, 'lineno', None)
                            })
            
            if var_names:
                # Deduplicate while preserving order
                seen = set()
                unique_vars = []
                for var in var_names:
                    if var['name'] not in seen:
                        seen.add(var['name'])
                        unique_vars.append(var)
                
                spec['variable_names'] = unique_vars
                spec['variable_names_critical'] = True  # Flag to emphasize importance
        
        except Exception:
            pass
        
        return spec
    
    def _refine_control_flow(
        self,
        specification: Dict[str, Any],
        target: RefinementTarget,
        original_code: str
    ) -> Dict[str, Any]:
        """Refine control flow documentation"""
        spec = specification.copy()
        
        try:
            tree = ast.parse(original_code)
            control_flow_details = []
            
            def extract_control_flow(node, depth=0):
                indent = "  " * depth
                if isinstance(node, ast.If):
                    control_flow_details.append(f"{indent}IF statement")
                    extract_control_flow(node.body, depth + 1)
                    if node.orelse:
                        control_flow_details.append(f"{indent}ELSE")
                        extract_control_flow(node.orelse, depth + 1)
                elif isinstance(node, ast.For):
                    control_flow_details.append(f"{indent}FOR loop")
                    extract_control_flow(node.body, depth + 1)
                elif isinstance(node, ast.While):
                    control_flow_details.append(f"{indent}WHILE loop")
                    extract_control_flow(node.body, depth + 1)
                elif isinstance(node, ast.Try):
                    control_flow_details.append(f"{indent}TRY block")
                    extract_control_flow(node.body, depth + 1)
                    for handler in node.handlers:
                        control_flow_details.append(f"{indent}EXCEPT")
                        extract_control_flow(handler.body, depth + 1)
                elif isinstance(node, list):
                    for item in node:
                        extract_control_flow(item, depth)
                elif hasattr(node, 'body'):
                    extract_control_flow(node.body, depth)
            
            funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            if funcs:
                extract_control_flow(funcs[0])
                spec['control_flow'] = "\n".join(control_flow_details)
        
        except Exception:
            pass
        
        return spec
    
    def _refine_edge_cases(
        self,
        specification: Dict[str, Any],
        target: RefinementTarget
    ) -> Dict[str, Any]:
        """Refine edge case documentation"""
        spec = specification.copy()
        
        # Add specific edge cases from issues
        edge_cases = spec.get('edge_cases', [])
        for issue in target.specific_issues:
            if 'not covered' in issue.lower():
                edge_case = {
                    'reference': 'coverage_gap',
                    'details': issue,
                    'priority': 'high'
                }
                if not any(ec.get('details') == issue for ec in edge_cases):
                    edge_cases.append(edge_case)
        
        spec['edge_cases'] = edge_cases
        return spec

