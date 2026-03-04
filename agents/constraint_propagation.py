"""
Constraint Propagation System
Propagates constraints from specifications to code regeneration
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import ast
import re
from dataclasses import dataclass
from enum import Enum


class ConstraintType(Enum):
    """Types of constraints"""
    VARIABLE_NAME = "variable_name"
    CONTROL_FLOW = "control_flow"
    RETURN_TYPE = "return_type"
    EXCEPTION = "exception"
    SIDE_EFFECT = "side_effect"
    DATA_STRUCTURE = "data_structure"
    ALGORITHM = "algorithm"


@dataclass
class Constraint:
    """A constraint that must be satisfied in regenerated code"""
    type: ConstraintType
    value: Any
    severity: str  # 'critical', 'important', 'suggested'
    source: str  # Where this constraint came from
    description: str


class ConstraintPropagator:
    """Propagates and enforces constraints during code regeneration"""
    
    def __init__(self):
        self.constraints: Dict[str, List[Constraint]] = {}
    
    def extract_constraints(
        self,
        specification: Dict[str, Any],
        original_code: str
    ) -> List[Constraint]:
        """Extract constraints from specification and original code"""
        constraints = []
        
        # Variable name constraints
        var_names = specification.get('variable_names', [])
        if var_names:
            for var in var_names:
                if isinstance(var, dict):
                    name = var.get('name', '')
                    if name and var.get('exact_match_required', False):
                        constraints.append(Constraint(
                            type=ConstraintType.VARIABLE_NAME,
                            value=name,
                            severity='critical',
                            source='specification',
                            description=f"Variable '{name}' must be used with exact spelling"
                        ))
        
        # Function signature constraints
        signature = specification.get('signature', {})
        # Handle case where signature might be a string (from slice-by-slice merging)
        if isinstance(signature, str):
            signature = {'raw': signature, 'parameters': []}
        elif not isinstance(signature, dict):
            signature = {}
        params = signature.get('parameters', [])
        for param in params:
            if isinstance(param, dict) and param.get('exact_match_required', False):
                param_name = param.get('name', '')
                if param_name:
                    constraints.append(Constraint(
                        type=ConstraintType.VARIABLE_NAME,
                        value=param_name,
                        severity='critical',
                        source='specification',
                        description=f"Parameter '{param_name}' must match exactly"
                    ))
        
        # Control flow constraints
        control_flow = specification.get('control_flow', '')
        if control_flow:
            constraints.append(Constraint(
                type=ConstraintType.CONTROL_FLOW,
                value=control_flow,
                severity='important',
                source='specification',
                description="Control flow structure must match"
            ))
        
        # Return type constraints
        return_type = specification.get('return_type', '')
        if return_type:
            constraints.append(Constraint(
                type=ConstraintType.RETURN_TYPE,
                value=return_type,
                severity='critical',
                source='specification',
                description=f"Return type must be {return_type}"
            ))
        
        # Exception constraints
        error_handling = specification.get('error_handling', {})
        if isinstance(error_handling, dict):
            exceptions = error_handling.get('exceptions', [])
            for exc in exceptions:
                if isinstance(exc, dict):
                    exc_type = exc.get('type', '')
                    if exc_type:
                        constraints.append(Constraint(
                            type=ConstraintType.EXCEPTION,
                            value=exc_type,
                            severity='important',
                            source='specification',
                            description=f"Must handle {exc_type} exception"
                        ))
        
        # Side effect constraints
        side_effects = specification.get('side_effects', {})
        if isinstance(side_effects, dict):
            has_side_effects = side_effects.get('has_side_effects', False)
            if not has_side_effects:
                constraints.append(Constraint(
                    type=ConstraintType.SIDE_EFFECT,
                    value=False,
                    severity='critical',
                    source='specification',
                    description="Function must have no side effects"
                ))
        
        # Extract constraints from original code AST
        code_constraints = self._extract_constraints_from_code(original_code)
        constraints.extend(code_constraints)
        
        return constraints
    
    def _extract_constraints_from_code(self, code: str) -> List[Constraint]:
        """Extract implicit constraints from original code"""
        constraints = []
        
        try:
            tree = ast.parse(code)
            
            # Extract variable names as constraints
            var_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_names.add(target.id)
            
            # Extract control flow structure
            control_flow_nodes = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                    control_flow_nodes.append(type(node).__name__)
            
            # Extract return statements
            return_statements = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Return):
                    if node.value:
                        return_statements.append(type(node.value).__name__)
            
            # Create constraints
            for var_name in var_names:
                if not var_name.startswith('_'):  # Skip private variables
                    constraints.append(Constraint(
                        type=ConstraintType.VARIABLE_NAME,
                        value=var_name,
                        severity='suggested',
                        source='code_analysis',
                        description=f"Variable '{var_name}' appears in original code"
                    ))
        
        except Exception:
            pass
        
        return constraints
    
    def generate_constraint_prompt_section(
        self,
        constraints: List[Constraint]
    ) -> str:
        """Generate prompt section enforcing constraints"""
        if not constraints:
            return ""
        
        # Group constraints by type
        by_type = {}
        for constraint in constraints:
            if constraint.type not in by_type:
                by_type[constraint.type] = []
            by_type[constraint.type].append(constraint)
        
        sections = []
        sections.append("\n\n" + "="*70)
        sections.append("CRITICAL CONSTRAINTS (MUST BE FOLLOWED):")
        sections.append("="*70)
        
        # Critical constraints first
        critical = [c for c in constraints if c.severity == 'critical']
        if critical:
            sections.append("\nCRITICAL CONSTRAINTS:")
            for constraint in critical:
                sections.append(f"  • {constraint.description}")
                sections.append(f"    Type: {constraint.type.value}")
                sections.append(f"    Value: {constraint.value}")
        
        # Important constraints
        important = [c for c in constraints if c.severity == 'important']
        if important:
            sections.append("\nIMPORTANT CONSTRAINTS:")
            for constraint in important:
                sections.append(f"  • {constraint.description}")
        
        # Variable name constraints (consolidated)
        if ConstraintType.VARIABLE_NAME in by_type:
            var_constraints = by_type[ConstraintType.VARIABLE_NAME]
            var_names = [c.value for c in var_constraints if isinstance(c.value, str)]
            if var_names:
                sections.append(f"\nVARIABLE NAME CONSTRAINTS:")
                sections.append(f"  You MUST use these exact variable names: {', '.join(var_names[:15])}")
        
        sections.append("\n" + "="*70)
        sections.append("Violating these constraints will result in low similarity scores.")
        sections.append("="*70)
        
        return "\n".join(sections)
    
    def validate_regenerated_code(
        self,
        constraints: List[Constraint],
        regenerated_code: str
    ) -> Tuple[bool, List[str]]:
        """Validate that regenerated code satisfies constraints"""
        violations = []
        
        try:
            tree = ast.parse(regenerated_code)
            
            # Check variable name constraints
            var_name_constraints = [c for c in constraints if c.type == ConstraintType.VARIABLE_NAME and c.severity == 'critical']
            if var_name_constraints:
                required_names = {c.value for c in var_name_constraints}
                actual_names = set()
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name):
                        actual_names.add(node.id)
                
                missing_names = required_names - actual_names
                if missing_names:
                    violations.append(f"Missing required variable names: {', '.join(missing_names)}")
            
            # Check control flow constraints
            control_flow_constraints = [c for c in constraints if c.type == ConstraintType.CONTROL_FLOW]
            for constraint in control_flow_constraints:
                # Simplified check - could be more sophisticated
                if constraint.value:
                    flow_text = str(constraint.value).lower()
                    code_text = regenerated_code.lower()
                    if 'if' in flow_text and 'if' not in code_text:
                        violations.append("Missing if statements required by control flow constraint")
            
            # Check return type constraints
            return_type_constraints = [c for c in constraints if c.type == ConstraintType.RETURN_TYPE]
            # Could add AST-based return type checking here
        
        except SyntaxError as e:
            violations.append(f"Syntax error in regenerated code: {e}")
        
        return len(violations) == 0, violations
    
    def propagate_to_prompt(
        self,
        base_prompt: str,
        constraints: List[Constraint]
    ) -> str:
        """Add constraint enforcement to prompt"""
        constraint_section = self.generate_constraint_prompt_section(constraints)
        return base_prompt + constraint_section

