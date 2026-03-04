"""
Specification Validator
Validates specifications before code regeneration to catch issues early
"""

from typing import Dict, List, Any, Optional, Tuple
import ast
import json
from dataclasses import dataclass


@dataclass
class ValidationIssue:
    """A validation issue found in specification"""
    severity: str  # 'error', 'warning', 'info'
    category: str  # 'completeness', 'consistency', 'accuracy'
    message: str
    field: Optional[str] = None


class SpecificationValidator:
    """Validates specifications for completeness, consistency, and accuracy"""
    
    def validate(
        self,
        specification: Dict[str, Any],
        original_code: str,
        function_name: str
    ) -> Tuple[bool, List[ValidationIssue]]:
        """
        Validate specification
        Returns: (is_valid, issues)
        """
        issues = []
        
        # Completeness checks
        issues.extend(self._check_completeness(specification, original_code, function_name))
        
        # Consistency checks
        issues.extend(self._check_consistency(specification, original_code))
        
        # Accuracy checks
        issues.extend(self._check_accuracy(specification, original_code))
        
        # Critical issues indicate invalidity
        has_errors = any(issue.severity == 'error' for issue in issues)
        
        return not has_errors, issues
    
    def _check_completeness(
        self,
        specification: Dict[str, Any],
        original_code: str,
        function_name: str
    ) -> List[ValidationIssue]:
        """Check if specification is complete"""
        issues = []
        
        # Check required fields
        required_fields = ['english_summary', 'signature']
        for field in required_fields:
            if field not in specification or not specification[field]:
                issues.append(ValidationIssue(
                    severity='error',
                    category='completeness',
                    message=f"Missing required field: {field}",
                    field=field
                ))
        
        # Check if signature has parameters
        signature = specification.get('signature', {})
        if isinstance(signature, dict):
            params = signature.get('parameters', [])
            if not params:
                # Check if function actually has parameters
                try:
                    tree = ast.parse(original_code)
                    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                    if funcs:
                        func = funcs[0]
                        non_self_params = [arg for arg in func.args.args if arg.arg != 'self']
                        if non_self_params:
                            issues.append(ValidationIssue(
                                severity='warning',
                                category='completeness',
                                message=f"Function has {len(non_self_params)} parameters but spec doesn't document them",
                                field='signature.parameters'
                            ))
                except Exception:
                    pass
        
        # Check if return type is documented for functions that return
        try:
            tree = ast.parse(original_code)
            has_returns = any(isinstance(n, ast.Return) for n in ast.walk(tree))
            if has_returns and not specification.get('return_type') and not specification.get('return_value'):
                issues.append(ValidationIssue(
                    severity='warning',
                    category='completeness',
                    message="Function has return statements but return type/value not documented",
                    field='return_type'
                ))
        except Exception:
            pass
        
        return issues
    
    def _check_consistency(
        self,
        specification: Dict[str, Any],
        original_code: str
    ) -> List[ValidationIssue]:
        """Check consistency between specification and code"""
        issues = []
        
        try:
            tree = ast.parse(original_code)
            
            # Check parameter count consistency
            signature = specification.get('signature', {})
            if isinstance(signature, dict):
                spec_params = signature.get('parameters', [])
                if isinstance(spec_params, list):
                    spec_param_count = len([p for p in spec_params if isinstance(p, dict) and p.get('name') != 'self'])
                    
                    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                    if funcs:
                        func = funcs[0]
                        code_param_count = len([arg for arg in func.args.args if arg.arg != 'self'])
                        
                        if spec_param_count != code_param_count:
                            issues.append(ValidationIssue(
                                severity='error',
                                category='consistency',
                                message=f"Parameter count mismatch: spec has {spec_param_count}, code has {code_param_count}",
                                field='signature.parameters'
                            ))
            
            # Check side effects consistency
            side_effects = specification.get('side_effects', {})
            if isinstance(side_effects, dict):
                claims_no_side_effects = side_effects.get('has_side_effects', True) == False
                
                # Check if code actually has side effects
                has_mutations = any(
                    isinstance(node, ast.Assign) and
                    isinstance(node.targets[0], ast.Attribute) if node.targets else False
                    for node in ast.walk(tree)
                )
                
                if claims_no_side_effects and has_mutations:
                    issues.append(ValidationIssue(
                        severity='error',
                        category='consistency',
                        message="Spec claims no side effects but code has state mutations",
                        field='side_effects'
                    ))
        
        except Exception:
            pass
        
        return issues
    
    def _check_accuracy(
        self,
        specification: Dict[str, Any],
        original_code: str
    ) -> List[ValidationIssue]:
        """Check accuracy of specification details"""
        issues = []
        
        # Check variable names against code
        spec_var_names = specification.get('variable_names', [])
        if spec_var_names:
            try:
                tree = ast.parse(original_code)
                code_vars = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                code_vars.add(target.id)
                
                spec_vars = set()
                for var in spec_var_names:
                    if isinstance(var, dict):
                        spec_vars.add(var.get('name', ''))
                    elif isinstance(var, str):
                        spec_vars.add(var)
                
                spec_vars.discard('')
                
                # Check for variables in spec but not in code
                extra_vars = spec_vars - code_vars
                if extra_vars:
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='accuracy',
                        message=f"Variables documented in spec but not found in code: {', '.join(list(extra_vars)[:5])}",
                        field='variable_names'
                    ))
            except Exception:
                pass
        
        return issues
    
    def generate_validation_feedback(self, issues: List[ValidationIssue]) -> str:
        """Generate feedback string from validation issues"""
        if not issues:
            return ""
        
        feedback_parts = []
        
        errors = [i for i in issues if i.severity == 'error']
        warnings = [i for i in issues if i.severity == 'warning']
        
        if errors:
            feedback_parts.append("VALIDATION ERRORS (must fix):")
            for issue in errors:
                feedback_parts.append(f"  ERROR: {issue.message} (field: {issue.field or 'N/A'})")
        
        if warnings:
            feedback_parts.append("\nVALIDATION WARNINGS:")
            for issue in warnings:
                feedback_parts.append(f"  WARNING: {issue.message} (field: {issue.field or 'N/A'})")
        
        return "\n".join(feedback_parts)

