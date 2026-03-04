"""
Failure-driven spec refinement.
Analyzes diff between original and regenerated code to infer missing abstract
specification content. Adds natural language updates (no code snippets) to improve
spec quality and regeneration accuracy.
"""

from typing import Dict, List, Any, Optional
from utils.code_diff_analyzer import CodeDiffAnalyzer


class FailureDrivenRefinementEngine:
    """
    Infers missing specification content from code differences.
    Uses diff analysis to generate abstract spec updates.
    """

    def __init__(self, call_llm):
        self.call_llm = call_llm
        self.diff_analyzer = CodeDiffAnalyzer()

    def analyze_diff_and_infer_spec_updates(
        self,
        original_code: str,
        regenerated_code: str,
        function_name: str,
        current_spec: Dict[str, Any],
        similarity_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Analyze diff and infer abstract spec updates (no code snippets).
        Returns dict with keys: abstract_updates, postconditions_to_add,
        preconditions_to_add, description_additions, etc.
        """
        if not original_code or not regenerated_code:
            return {'abstract_updates': [], 'success': False}

        diff_result = self.diff_analyzer.get_code_differences(original_code, regenerated_code)
        missing_lines = diff_result.get('missing_lines', [])
        diff_blocks = diff_result.get('diff_blocks', [])

        if not missing_lines and not diff_blocks:
            return {'abstract_updates': [], 'success': True}

        # Build context for LLM
        missing_context = '\n'.join(missing_lines[:20]) if missing_lines else 'None'
        block_summaries = []
        for i, block in enumerate(diff_blocks[:5]):
            orig = block.get('original', [])
            regen = block.get('regenerated', [])
            block_summaries.append(
                f"Block {i+1}: Original had {len(orig)} lines, regenerated had {len(regen)}"
            )

        prompt = f"""You are a specification refinement expert. The following function was regenerated from a specification, but the regenerated code DIFFERS from the original.

FUNCTION: {function_name}

ORIGINAL CODE (excerpt - lines present in original but missing in regenerated):
```
{missing_context}
```

CURRENT SPECIFICATION (excerpt):
- Description: {str(current_spec.get('description', ''))[:300]}
- Postconditions: {str(current_spec.get('postconditions', []))[:200]}

SIMILARITY GAPS:
- Structural: {similarity_metrics.get('structural_similarity', 0):.1%}
- Behavioral: {similarity_metrics.get('behavioral_similarity', 0):.1%}
- Textual: {similarity_metrics.get('textual_similarity', 0):.1%}

TASK: Infer what ABSTRACT specification content is missing. Provide natural language additions only - NO code snippets.
Output a JSON object with these optional keys:
{{
  "postconditions_to_add": ["list of postcondition strings to add"],
  "preconditions_to_add": ["list of precondition strings to add"],
  "description_additions": "paragraph to append to description",
  "edge_cases_to_document": ["edge case descriptions"],
  "variable_constraints": ["constraints on variables that may be missing"]
}}

Only include keys for which you have concrete suggestions. Be specific and actionable.
Output valid JSON only, no markdown."""

        try:
            response = self.call_llm(prompt)
            return self._parse_and_validate_response(response, current_spec)
        except Exception as e:
            return {'abstract_updates': [], 'success': False, 'error': str(e)}

    def _parse_and_validate_response(
        self, response: str, current_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse LLM response and validate structure."""
        import json
        import re

        response = response.strip()
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if not json_match:
            return {'abstract_updates': [], 'success': False}

        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return {'abstract_updates': [], 'success': False}

        result = {
            'postconditions_to_add': parsed.get('postconditions_to_add', []),
            'preconditions_to_add': parsed.get('preconditions_to_add', []),
            'description_additions': parsed.get('description_additions', ''),
            'edge_cases_to_document': parsed.get('edge_cases_to_document', []),
            'variable_constraints': parsed.get('variable_constraints', []),
            'success': True
        }
        return result

    def merge_updates_into_spec(
        self, spec: Dict[str, Any], updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge abstract updates into specification. Modifies spec in place, returns it."""
        if not updates.get('success'):
            return spec

        postconditions = spec.get('postconditions', [])
        if isinstance(postconditions, str):
            postconditions = [postconditions] if postconditions else []
        for item in updates.get('postconditions_to_add', []):
            if item and item not in postconditions:
                postconditions.append(item)
        spec['postconditions'] = postconditions

        preconditions = spec.get('preconditions', [])
        if isinstance(preconditions, str):
            preconditions = [preconditions] if preconditions else []
        for item in updates.get('preconditions_to_add', []):
            if item and item not in preconditions:
                preconditions.append(item)
        spec['preconditions'] = preconditions

        desc_add = updates.get('description_additions', '').strip()
        if desc_add:
            existing = spec.get('description', '')
            spec['description'] = f"{existing}\n\nRefinement: {desc_add}".strip()

        edge_cases = spec.get('edge_cases', [])
        if isinstance(edge_cases, str):
            edge_cases = [edge_cases] if edge_cases else []
        for item in updates.get('edge_cases_to_document', []):
            if item and item not in edge_cases:
                edge_cases.append(item)
        spec['edge_cases'] = edge_cases

        var_constraints = updates.get('variable_constraints', [])
        if var_constraints:
            if 'variable_constraints' not in spec:
                spec['variable_constraints'] = []
            for c in var_constraints:
                if c and c not in spec['variable_constraints']:
                    spec['variable_constraints'].append(c)

        if 'failure_driven_updates' not in spec:
            spec['failure_driven_updates'] = []
        spec['failure_driven_updates'].append(updates)

        return spec
