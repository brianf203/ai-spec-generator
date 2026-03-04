"""
Logical deletion planning.
Identifies code statements that can be safely removed when they do not
contribute to any slicing criterion or causal minimal element.
Uses causal inference to avoid deleting causally important code.
"""

from typing import Dict, Any, List, Set, Optional
import textwrap
import re


class LogicalDeletionPass:
    """Plan logical deletions and assertion targets for each slice."""

    def build_plan(
        self,
        source_code: str,
        slicing_analysis: Dict[str, Any],
        specification: Dict[str, Any],
        function_name: str = ''
    ) -> Dict[str, Any]:
        if not slicing_analysis or not slicing_analysis.get('slices'):
            return {}
        if not source_code:
            return {}

        source_lines = textwrap.dedent(source_code).splitlines()
        slices = slicing_analysis.get('slices', [])

        critical_lines = self._collect_critical_lines(slices, specification)
        critical_lines.update(self._get_causal_critical_lines(source_code, function_name))
        covered_lines = self._collect_covered_lines(slices)

        if not covered_lines:
            return {}

        deletable_lines = sorted(line for line in covered_lines if line not in critical_lines)
        deletable_snippets = [self._line_to_snippet(source_lines, line) for line in deletable_lines]
        critical_snippets = [self._line_to_snippet(source_lines, line) for line in sorted(critical_lines)]

        slice_assertions = self._build_slice_assertions(slices)

        return {
            'critical_lines': sorted(critical_lines),
            'critical_snippets': [snippet for snippet in critical_snippets if snippet],
            'deletable_lines': deletable_lines,
            'deletable_snippets': [snippet for snippet in deletable_snippets if snippet],
            'slice_assertions': slice_assertions,
            'summary': self._summarize_plan(critical_lines, deletable_lines, slice_assertions)
        }

    def _collect_critical_lines(
        self,
        slices: List[Dict[str, Any]],
        specification: Dict[str, Any]
    ) -> Set[int]:
        critical = set()
        for slice_info in slices:
            critical.update(slice_info.get('statement_lines', []))
            critical.update(slice_info.get('guard_lines', []))
        for elem in specification.get('minimal_elements', []):
            if isinstance(elem, str):
                match = re.match(r"L(\d+)", elem)
                if match:
                    critical.add(int(match.group(1)))
            elif isinstance(elem, int):
                critical.add(elem)
        return critical

    def _get_causal_critical_lines(self, source_code: str, function_name: str) -> Set[int]:
        """Use causal inference to identify lines that must not be deleted."""
        critical = set()
        try:
            from agents.causal_inference import CausalSpecificationInferencer
            inferencer = CausalSpecificationInferencer(source_code, function_name)
            spec = inferencer.generate_specification_from_causal_analysis()
            for elem in spec.get('minimal_elements', []):
                if isinstance(elem, str):
                    match = re.match(r"L(\d+)", elem)
                    if match:
                        critical.add(int(match.group(1)))
                elif isinstance(elem, int):
                    critical.add(elem)
        except Exception:
            pass
        return critical

    def _collect_covered_lines(self, slices: List[Dict[str, Any]]) -> Set[int]:
        covered = set()
        for slice_info in slices:
            start, end = slice_info.get('line_range', (0, 0))
            if start and end and end >= start:
                covered.update(range(start, end + 1))
        return covered

    def _build_slice_assertions(self, slices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        assertions = []
        for slice_info in slices:
            slice_id = slice_info.get('slice_id') or slice_info.get('criterion', {}).get('description', 'slice')
            guard_conditions = slice_info.get('guard_conditions') or []
            guard_text = ' AND '.join(guard_conditions[:3]) if guard_conditions else 'default path'
            criterion = slice_info.get('criterion', {})
            expected = criterion.get('description') or criterion.get('target_code') or 'documented effect'
            assertions.append({
                'assertion_id': f"assert_{slice_id}",
                'slice_id': slice_id,
                'precondition': guard_text,
                'expected_effect': expected,
                'line_range': slice_info.get('line_range', (0, 0))
            })
        return assertions

    def _line_to_snippet(self, source_lines: List[str], line_no: int) -> str:
        if not line_no:
            return ''
        index = line_no - 1
        if index < 0 or index >= len(source_lines):
            return ''
        return source_lines[index].rstrip()

    def _summarize_plan(
        self,
        critical_lines: Set[int],
        deletable_lines: List[int],
        slice_assertions: List[Dict[str, Any]]
    ) -> str:
        return (
            f"Critical lines retained: {len(critical_lines)} | "
            f"Lines flagged for deletion: {len(deletable_lines)} | "
            f"Slice assertions: {len(slice_assertions)}"
        )

