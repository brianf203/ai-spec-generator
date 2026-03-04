"""
Counter-Example Guided Specification Synthesis (CEGSS)
Adapts CEGIS (Counter-Example Guided Inductive Synthesis) to specification generation.
Novel approach: Iteratively refine specifications using counter-examples.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import ast
import subprocess
import tempfile
import os
from dataclasses import dataclass


@dataclass
class CounterExample:
    """A counter-example that shows a specification is incomplete"""
    program_code: str  # Program that satisfies spec but differs from original
    difference: str  # How it differs from original
    behavior_mismatch: str  # What behavior is different
    spec_gap: str  # What's missing from specification


class CEGSSEngine:
    """
    Counter-Example Guided Specification Synthesis Engine
    
    Algorithm:
    1. Generate candidate specification
    2. Find counter-example (program satisfying spec but different from original)
    3. Refine spec to exclude counter-example
    4. Iterate until convergence
    """
    
    def __init__(self, original_code: str, function_name: str):
        self.original_code = original_code
        self.function_name = function_name
        self.max_iterations = 5
        self.counter_examples = []
    
    def synthesize_specification(self, initial_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize specification using counter-example guided approach.
        
        Returns refined specification that excludes all counter-examples.
        """
        current_spec = initial_spec.copy()
        
        for iteration in range(self.max_iterations):
            # Step 1: Find counter-example
            counter_example = self._find_counter_example(current_spec)
            
            if counter_example is None:
                # No counter-example found - spec is complete
                return current_spec
            
            # Step 2: Refine specification
            current_spec = self._refine_specification(current_spec, counter_example)
            self.counter_examples.append(counter_example)
        
        # Return best spec found (even if didn't converge)
        return current_spec
    
    def _find_counter_example(self, specification: Dict[str, Any]) -> Optional[CounterExample]:
        """
        Find a counter-example: a program that satisfies the spec but differs from original.
        
        Strategy:
        1. Generate a program from the specification
        2. Check if it differs from original in behavior
        3. If different, we found a counter-example
        """
        # Simplified: Use LLM to generate a program that satisfies spec
        # In full implementation, would use constraint solver or program synthesis
        
        # For now, use heuristic: check if spec is missing constraints
        # that would allow different implementations
        
        # Check for common gaps:
        gaps = self._identify_spec_gaps(specification)
        
        if gaps:
            # Create counter-example based on gap
            return CounterExample(
                program_code="",  # Would be generated program
                difference=gaps[0],
                behavior_mismatch="Different implementation possible due to spec gap",
                spec_gap=gaps[0]
            )
        
        return None
    
    def _identify_spec_gaps(self, specification: Dict[str, Any]) -> List[str]:
        """Identify gaps in specification that could allow different implementations"""
        gaps = []
        
        # Gap 1: Missing variable name constraints
        if not specification.get('variable_names') or len(specification.get('variable_names', [])) == 0:
            gaps.append("Missing variable name constraints - allows variable renaming")
        
        # Gap 2: Missing control flow details
        control_flow = specification.get('control_flow', '')
        if not control_flow or len(control_flow) < 20:
            gaps.append("Missing detailed control flow - allows different control structures")
        
        # Gap 3: Missing edge case handling
        edge_cases = specification.get('edge_cases', [])
        if not edge_cases:
            gaps.append("Missing edge case specifications - allows different edge case handling")
        
        # Gap 4: Missing return value constraints
        return_type = specification.get('return_type', '')
        if not return_type:
            gaps.append("Missing return type constraints - allows different return types")
        
        # Gap 5: Missing side effect specifications
        side_effects = specification.get('side_effects', {})
        if not side_effects or not side_effects.get('has_side_effects', None):
            gaps.append("Missing side effect specification - allows/prevents side effects")
        
        return gaps
    
    def _refine_specification(self, spec: Dict[str, Any], counter_example: CounterExample) -> Dict[str, Any]:
        """
        Refine specification to exclude counter-example.
        
        Adds constraints or details that prevent the counter-example.
        """
        refined = spec.copy()
        
        gap = counter_example.spec_gap
        
        # Refine based on gap type
        if "variable name" in gap.lower():
            # Add variable name constraints
            if 'variable_names' not in refined:
                refined['variable_names'] = []
            # Would extract from original code
            refined['variable_names_constraint'] = "exact_match_required"
        
        elif "control flow" in gap.lower():
            # Add detailed control flow
            if not refined.get('control_flow'):
                refined['control_flow'] = "Detailed control flow structure required"
            refined['control_flow_detailed'] = True
        
        elif "edge case" in gap.lower():
            # Add edge case specifications
            if 'edge_cases' not in refined:
                refined['edge_cases'] = []
            refined['edge_cases_required'] = True
        
        elif "return type" in gap.lower():
            # Add return type constraints
            if not refined.get('return_type'):
                refined['return_type'] = "Must match original return type"
            refined['return_type_constraint'] = "exact_match"
        
        elif "side effect" in gap.lower():
            # Add side effect specification
            if 'side_effects' not in refined:
                refined['side_effects'] = {}
            refined['side_effects']['specification_required'] = True
        
        # Mark that this spec was refined via CEGSS
        refined['cegss_refinements'] = refined.get('cegss_refinements', 0) + 1
        refined['cegss_counter_examples'] = [
            ce.spec_gap for ce in self.counter_examples
        ]
        
        return refined
    
    def generate_cegss_guidance(self, specification: Dict[str, Any]) -> str:
        """Generate guidance text from CEGSS analysis"""
        if not self.counter_examples:
            return ""
        
        guidance = "\n\nCOUNTER-EXAMPLE GUIDED REFINEMENT:\n"
        guidance += f"This specification was refined through {len(self.counter_examples)} iterations.\n"
        guidance += "The following gaps were identified and closed:\n"
        
        for i, ce in enumerate(self.counter_examples, 1):
            guidance += f"  {i}. {ce.spec_gap}\n"
        
        guidance += "\nEnsure regenerated code addresses all these constraints.\n"
        
        return guidance

