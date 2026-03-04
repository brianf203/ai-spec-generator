"""
Slice-by-Slice Specification Generator
1. Extract code for each slice
2. Generate spec for each slice independently
3. Filter specs using logical deletion
4. Merge all slice specs into one complete spec
"""

from typing import Dict, List, Any, Optional, Callable
import ast
import textwrap
import json
import re


class SliceBySliceSpecGenerator:
    """
    Generates specifications slice-by-slice, then merges them.
    """
    
    def __init__(self, call_llm: Callable[[str], str]):
        self.call_llm = call_llm
    
    def generate_spec_from_slices(
        self,
        source_code: str,
        slicing_analysis: Dict[str, Any],
        function_name: str,
        func_info: Dict[str, Any],
        causal_minimal_elements: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate specification using slice-by-slice approach.
        
        Returns:
            Dictionary with 'success' flag and either 'specification' or 'error'
        """
        if not slicing_analysis or not slicing_analysis.get('slices'):
            return {
                'success': False,
                'error': 'No slices available for slice-by-slice generation'
            }
        
        slices = slicing_analysis.get('slices', [])
        if len(slices) == 0:
            return {
                'success': False,
                'error': 'Empty slice list'
            }
        
        print(f"        Generating specs for {len(slices)} slices...")
        
        # Step 1: Extract code snippet for each slice
        source_lines = textwrap.dedent(source_code).splitlines()
        slice_code_snippets = []
        
        for slice_info in slices:
            slice_code = self._extract_slice_code(slice_info, source_lines, source_code)
            if slice_code:
                slice_code_snippets.append({
                    'slice_info': slice_info,
                    'code': slice_code
                })
        
        if not slice_code_snippets:
            return {
                'success': False,
                'error': 'Could not extract code for any slice'
            }
        
        # Step 2: Generate spec for each slice
        slice_specs = []
        for idx, slice_data in enumerate(slice_code_snippets, 1):
            slice_info = slice_data['slice_info']
            slice_code = slice_data['code']
            
            print(f"          Generating spec for slice {idx}/{len(slice_code_snippets)}...")
            slice_spec = self._generate_spec_for_slice(
                slice_code,
                slice_info,
                function_name,
                func_info,
                idx,
                len(slice_code_snippets)
            )
            
            if slice_spec:
                slice_specs.append({
                    'slice_info': slice_info,
                    'spec': slice_spec,
                    'slice_code': slice_code
                })
            else:
                print(f"          WARNING: Failed to generate spec for slice {idx}")
        
        if not slice_specs:
            return {
                'success': False,
                'error': 'Failed to generate specs for any slice'
            }
        
        print(f"        Generated {len(slice_specs)}/{len(slice_code_snippets)} slice specs")
        
        # Step 3: Filter slice specs using logical deletion criteria
        filtered_slice_specs = self._filter_slice_specs(
            slice_specs,
            source_code,
            slicing_analysis
        )
        
        print(f"        Filtered to {len(filtered_slice_specs)}/{len(slice_specs)} valid specs")
        
        # Step 4: Merge all slice specs into one complete spec (with causal prioritization)
        merged_spec = self._merge_slice_specs(
            filtered_slice_specs,
            function_name,
            func_info,
            source_code,
            causal_minimal_elements
        )
        
        return {
            'success': True,
            'specification': merged_spec,
            'slice_specs': [s['spec'] for s in filtered_slice_specs],
            'num_slices': len(slices),
            'num_generated': len(slice_specs),
            'num_filtered': len(filtered_slice_specs)
        }
    
    def _extract_slice_code(
        self,
        slice_info: Dict[str, Any],
        source_lines: List[str],
        full_source: str
    ) -> Optional[str]:
        """Extract code snippet for a slice based on line range, including function signature"""
        # Try to extract function signature first
        func_signature = self._extract_function_signature(full_source)
        
        line_range = slice_info.get('line_range', (0, 0))
        statement_lines = slice_info.get('statement_lines', [])
        guard_lines = slice_info.get('guard_lines', [])
        
        if not statement_lines:
            # Fall back to line range
            start_line, end_line = line_range
            if start_line > 0 and end_line >= start_line:
                # Convert to 0-based indexing
                start_idx = max(0, start_line - 1)
                end_idx = min(len(source_lines), end_line)
                if start_idx < end_idx:
                    slice_body = '\n'.join(source_lines[start_idx:end_idx])
                    # Calculate nesting depth
                    nesting_depth = self._calculate_slice_nesting_depth(slice_body, full_source)
                    # Prepend function signature if available
                    if func_signature:
                        return func_signature + '\n    ' + slice_body.replace('\n', '\n    ')
                    return slice_body
            return None
        
        # Collect all relevant lines
        all_lines = set(statement_lines)
        all_lines.update(guard_lines)
        
        if not all_lines:
            return None
        
        min_line = min(all_lines)
        max_line = max(all_lines)
        
        # Extract context: include a few lines before and after
        context_before = 2
        context_after = 2
        
        start_idx = max(0, min_line - 1 - context_before)
        end_idx = min(len(source_lines), max_line + context_after)
        
        if start_idx >= end_idx:
            return None
        
        slice_body = '\n'.join(source_lines[start_idx:end_idx])
        
        # Calculate nesting depth for this slice
        nesting_depth = self._calculate_slice_nesting_depth(slice_body, full_source)
        
        # Prepend function signature if available (helps LLM understand context)
        if func_signature:
            # Preserve original indentation from source
            # Extract the actual indentation pattern from the slice
            slice_lines = slice_body.split('\n')
            if slice_lines:
                # Find base indentation (minimum indentation in slice)
                base_indent = min(len(line) - len(line.lstrip()) for line in slice_lines if line.strip()) if any(line.strip() for line in slice_lines) else 0
                # Preserve relative indentation
                indented_body = '\n'.join(
                    ('    ' * max(0, (len(line) - len(line.lstrip()) - base_indent) // 4) + line.lstrip()) if line.strip() else line
                    for line in slice_lines
                )
            else:
                indent = '    '
                indented_body = '\n'.join(
                    indent + line if line.strip() else line
                    for line in slice_body.split('\n')
                )
            return func_signature + '\n' + indented_body
        
        return slice_body
    
    def _calculate_slice_nesting_depth(self, slice_code: str, full_source: str) -> int:
        """Calculate maximum nesting depth in a slice"""
        try:
            import ast
            # Try to parse just the slice, but if it fails, parse full source and find relevant nodes
            try:
                tree = ast.parse(slice_code)
            except SyntaxError:
                # Slice might not be complete, parse full source instead
                tree = ast.parse(full_source)
            
            max_depth = 0
            
            def visit_node(node, depth=0):
                nonlocal max_depth
                max_depth = max(max_depth, depth)
                
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                        visit_node(child, depth + 1)
                    else:
                        visit_node(child, depth)
            
            visit_node(tree)
            return max_depth
        except Exception:
            # Fallback: count indentation levels
            lines = slice_code.split('\n')
            max_indent = 0
            for line in lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    max_indent = max(max_indent, indent)
            return max_indent // 4  # Assuming 4 spaces per level
    
    def _extract_function_signature(self, source_code: str) -> Optional[str]:
        """Extract function signature (def line) from source code"""
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    lines = source_code.splitlines()
                    # Get the def line
                    def_line_idx = node.lineno - 1
                    if 0 <= def_line_idx < len(lines):
                        def_line = lines[def_line_idx]
                        # Include docstring if present (first line after def)
                        result = def_line
                        if node.body and isinstance(node.body[0], ast.Expr):
                            if isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                                docstring_line = node.body[0].lineno - 1
                                if 0 <= docstring_line < len(lines):
                                    result += '\n    ' + lines[docstring_line]
                        return result
        except Exception:
            pass
        return None
    
    def _generate_spec_for_slice(
        self,
        slice_code: str,
        slice_info: Dict[str, Any],
        function_name: str,
        func_info: Dict[str, Any],
        slice_index: int,
        total_slices: int
    ) -> Optional[Dict[str, Any]]:
        """Generate specification for a single slice with retry logic"""
        
        criterion = slice_info.get('criterion', {})
        criterion_desc = criterion.get('description') or criterion.get('type') or 'unknown'
        variables = slice_info.get('variables', []) or []
        control_structures = slice_info.get('control_structures', []) or []
        guard_conditions = slice_info.get('guard_conditions', []) or []
        
        # Get original variable names from source code to preserve them
        original_source = func_info.get('source_code', '') or ''
        preserved_vars = self._extract_variable_names_from_code(original_source, variables) if original_source else []
        
        # Calculate nesting depth for this slice
        nesting_depth = self._calculate_slice_nesting_depth(slice_code, original_source)
        nesting_info = f"- Maximum nesting depth: {nesting_depth} levels (CRITICAL: preserve exact indentation levels)"
        
        prompt = f"""Generate a JSON specification for ONLY this code slice (slice {slice_index} of {total_slices}).

SLICE CONTEXT:
- Slice Criterion: {criterion_desc}
- Variables used: {', '.join(variables[:10])}
- Control structures: {', '.join(control_structures) if control_structures else 'none'}
- Guard conditions: {', '.join(guard_conditions[:3]) if guard_conditions else 'default path'}
{nesting_info}
{f'- CRITICAL: Preserve these exact variable names: {", ".join(preserved_vars[:5])}' if preserved_vars else ''}

CODE SLICE TO ANALYZE:
```python
{slice_code}
```

FUNCTION CONTEXT:
- Function name: {function_name}
- This is part of a larger function that has been decomposed into {total_slices} slices.

INSTRUCTIONS:
Generate a JSON specification that describes ONLY what this slice does. Focus on:
1. What output/effect this slice produces
2. What conditions must be true for this slice to execute
3. What variables this slice reads/writes (use EXACT variable names from code)
4. Any side effects
5. Control flow structure (if/else branches, loops, etc.)

CRITICAL REQUIREMENTS:
- Use EXACT variable names from the code (do NOT rename or use synonyms)
- Preserve control flow structure (if conditions, loop types, etc.)
- CRITICAL: Document nesting depth and indentation levels - this slice has {nesting_depth} levels of nesting
- Preserve exact indentation: each nested level must be indented by exactly 4 spaces more than its parent
- Be specific about conditions and logic

The specification should be a JSON object with these fields:
{{
    "slice_id": "slice_{slice_index}",
    "description": "Brief description of what this slice does",
    "preconditions": ["condition1", "condition2"],
    "postconditions": ["effect1", "effect2"],
    "variables_read": ["var1", "var2"],
    "variables_written": ["var1", "var2"],
    "return_value": "description if this slice produces a return value",
    "side_effects": ["side effect description"],
    "user_stories": [
        {{
            "id": "story_1",
            "as_a": "component",
            "i_want": "description",
            "so_that": "purpose"
        }}
    ],
    "test_cases": [
        {{
            "id": "test_1",
            "description": "test description",
            "preconditions": {{"var": "value"}},
            "expected_output": "expected result"
        }}
    ]
}}

IMPORTANT:
- Focus ONLY on this slice, not the entire function
- Be specific about what this slice contributes
- If this slice handles a specific condition/path, describe that condition
- Generate valid JSON only, no markdown or code blocks

Generate the JSON specification now:"""
        
        # Retry logic for slice spec generation
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self.call_llm(prompt)
                spec = self._parse_slice_spec_response(response)
                
                if spec:
                    # Add slice metadata
                    spec['slice_index'] = slice_index
                    spec['criterion'] = criterion_desc
                    spec['slice_code'] = slice_code
                    spec['nesting_depth'] = nesting_depth  # Store nesting depth for merging
                    
                    # Preserve variable names
                    if preserved_vars:
                        if 'variables_read' in spec:
                            spec['variables_read'] = list(set(spec['variables_read'] + preserved_vars))
                        if 'variables_written' in spec:
                            spec['variables_written'] = list(set(spec['variables_written'] + preserved_vars))
                    
                    return spec
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"          Retrying slice {slice_index} (attempt {attempt + 1}/{max_retries})...")
                else:
                    print(f"          ERROR generating spec for slice {slice_index}: {e}")
        
        return None
    
    def _extract_variable_names_from_code(self, source_code: str, slice_variables: List[str]) -> List[str]:
        """Extract actual variable names from source code to preserve them using AST"""
        preserved = []
        try:
            tree = ast.parse(source_code)
            # Extract all variable names from AST (more accurate than regex)
            code_vars = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    code_vars.add(node.id)
                elif isinstance(node, ast.FunctionDef):
                    # Include function parameters
                    for arg in node.args.args:
                        code_vars.add(arg.arg)
                elif isinstance(node, ast.Assign):
                    # Include assignment targets
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            code_vars.add(target.id)
                        elif isinstance(target, ast.Tuple):
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    code_vars.add(elt.id)
            
            # Match slice variables to code variables (case-insensitive, preserve exact case)
            slice_var_lower = {v.lower(): v for v in slice_variables}
            for code_var in code_vars:
                code_var_lower = code_var.lower()
                if code_var_lower in slice_var_lower:
                    # Preserve exact case from source code
                    preserved.append(code_var)
        except Exception:
            # Fallback to original method if AST parsing fails
            source_lower = source_code.lower()
            for var in slice_variables:
                var_lower = var.lower()
                if var_lower in source_lower:
                    for line in source_code.splitlines():
                        if var_lower in line.lower():
                            import re
                            matches = re.findall(r'\b' + re.escape(var) + r'\b', line, re.IGNORECASE)
                            if matches:
                                preserved.append(matches[0])
                                break
        
        return list(set(preserved))
    
    def _parse_slice_spec_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response to extract JSON specification"""
        # Try to extract JSON from response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith('```'):
            lines = response.split('\n')
            # Find the start and end of code block
            start_idx = 0
            end_idx = len(lines)
            for i, line in enumerate(lines):
                if line.strip().startswith('```json'):
                    start_idx = i + 1
                elif line.strip() == '```' and i > start_idx:
                    end_idx = i
                    break
            response = '\n'.join(lines[start_idx:end_idx])
        
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                spec = json.loads(json_str)
                return spec
            except json.JSONDecodeError:
                pass
        
        # Try parsing the whole response as JSON
        try:
            spec = json.loads(response)
            return spec
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _filter_slice_specs(
        self,
        slice_specs: List[Dict[str, Any]],
        source_code: str,
        slicing_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter slice specs using logical deletion criteria.
        Remove specs that:
        1. Don't describe meaningful behavior
        2. Don't match their slice's actual code
        3. Are duplicates or too similar
        """
        if not slice_specs:
            return []
        
        filtered = []
        seen_descriptions = set()  # Track duplicates
        
        for slice_data in slice_specs:
            slice_info = slice_data['slice_info']
            spec = slice_data['spec']
            slice_code = slice_data['slice_code']
            
            # Filter criteria
            is_valid = True
            
            # Check 1: Spec must have meaningful content
            if not spec.get('description') and not spec.get('user_stories'):
                is_valid = False
            
            # Check 2: Spec must mention variables from the slice
            spec_text = json.dumps(spec, default=str).lower()
            slice_vars = slice_info.get('variables', [])
            if slice_vars:
                var_mentioned = any(var.lower() in spec_text for var in slice_vars[:5])
                if not var_mentioned:
                    # Allow if spec is very short (might be a simple slice)
                    if len(slice_code.split('\n')) > 3:
                        is_valid = False
            
            # Check 3: Spec must match the criterion
            criterion = slice_info.get('criterion', {})
            criterion_desc = (criterion.get('description') or criterion.get('type') or '').lower()
            if criterion_desc and 'return' in criterion_desc:
                # Slice is about return value - spec should mention return
                if 'return' not in spec_text and 'output' not in spec_text:
                    is_valid = False
            
            # Check 4: Remove duplicates (same description)
            if is_valid:
                desc = spec.get('description', '').strip().lower()
                if desc and desc in seen_descriptions:
                    # Duplicate description - keep the one with more detail
                    existing_idx = next(
                        (i for i, f in enumerate(filtered) 
                         if f['spec'].get('description', '').strip().lower() == desc),
                        None
                    )
                    if existing_idx is not None:
                        existing = filtered[existing_idx]
                        # Keep the one with more content
                        if len(json.dumps(spec, default=str)) > len(json.dumps(existing['spec'], default=str)):
                            filtered[existing_idx] = slice_data
                        is_valid = False
                    else:
                        seen_descriptions.add(desc)
                elif desc:
                    seen_descriptions.add(desc)
            
            if is_valid:
                filtered.append(slice_data)
        
        return filtered
    
    def _merge_slice_specs(
        self,
        slice_specs: List[Dict[str, Any]],
        function_name: str,
        func_info: Dict[str, Any],
        source_code: str,
        causal_minimal_elements: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Merge multiple slice specs into one complete specification.
        Uses causal prioritization: slices containing causally important
        lines are processed first so their content gets priority.
        """
        if not slice_specs:
            return {}

        causal_lines = set()
        if causal_minimal_elements:
            for elem in causal_minimal_elements:
                if isinstance(elem, str):
                    m = re.match(r"L(\d+)", elem)
                    if m:
                        causal_lines.add(int(m.group(1)))
                elif isinstance(elem, int):
                    causal_lines.add(elem)

        if causal_lines:
            def causal_priority(slice_data: Dict[str, Any]) -> int:
                si = slice_data.get('slice_info', {})
                stmt_lines = set(si.get('statement_lines', []))
                guard_lines = set(si.get('guard_lines', []))
                start, end = si.get('line_range', (0, 0))
                range_lines = set(range(start, end + 1)) if end >= start else set()
                overlap = len((stmt_lines | guard_lines | range_lines) & causal_lines)
                return -overlap

            slice_specs = sorted(slice_specs, key=causal_priority)
        
        # Start with base specification structure
        # Ensure signature is always a dict, not a string
        signature = func_info.get('signature', {})
        if isinstance(signature, str):
            # If signature is a string, convert to dict format
            signature = {'raw': signature, 'parameters': []}
        elif not isinstance(signature, dict):
            signature = {}
        
        merged = {
            'function_name': function_name,
            'signature': signature,
            'description': '',
            'preconditions': [],
            'postconditions': [],
            'variables': [],
            'user_stories': [],
            'test_cases': [],
            'slice_specs': []
        }
        
        # Collect all user stories
        all_user_stories = []
        all_test_cases = []
        all_preconditions = []
        all_postconditions = []
        all_variables = set()
        max_nesting_depth = 0  # Track maximum nesting depth across all slices
        
        slice_descriptions = []
        
        for slice_data in slice_specs:
            spec = slice_data['spec']
            slice_info = slice_data['slice_info']
            
            # Add slice-specific info
            merged['slice_specs'].append({
                'slice_id': slice_info.get('slice_id'),
                'description': spec.get('description', ''),
                'criterion': slice_info.get('criterion', {})
            })
            
            # Collect descriptions
            if spec.get('description'):
                slice_descriptions.append(f"- {spec['description']}")
            
            # Merge user stories
            if spec.get('user_stories'):
                for story in spec['user_stories']:
                    # Ensure story is a dict
                    if isinstance(story, dict):
                        # Add slice context to story
                        story = story.copy()  # Don't modify original
                        story['slice_id'] = slice_info.get('slice_id')
                        all_user_stories.append(story)
                    elif isinstance(story, str):
                        # Convert string to dict format
                        all_user_stories.append({
                            'id': f"slice_{slice_info.get('slice_id', 'unknown')}_story",
                            'title': story,
                            'narrative': story,
                            'slice_id': slice_info.get('slice_id')
                        })
            
            # Merge test cases
            if spec.get('test_cases'):
                for test in spec['test_cases']:
                    # Ensure test is a dict
                    if isinstance(test, dict):
                        test = test.copy()  # Don't modify original
                        test['slice_id'] = slice_info.get('slice_id')
                        all_test_cases.append(test)
                    elif isinstance(test, str):
                        # Convert string to dict format
                        all_test_cases.append({
                            'id': f"slice_{slice_info.get('slice_id', 'unknown')}_test",
                            'description': test,
                            'slice_id': slice_info.get('slice_id')
                        })
            
            # Merge preconditions/postconditions
            if spec.get('preconditions'):
                all_preconditions.extend(spec['preconditions'])
            
            if spec.get('postconditions'):
                all_postconditions.extend(spec['postconditions'])
            
            # Merge variables - preserve exact names from source code
            if spec.get('variables_read'):
                for var in spec['variables_read']:
                    if isinstance(var, str):
                        all_variables.add(var)
            if spec.get('variables_written'):
                for var in spec['variables_written']:
                    if isinstance(var, str):
                        all_variables.add(var)
            
            # Extract control flow from slice info if available
            if slice_info.get('control_structures'):
                if 'control_flow' not in merged:
                    merged['control_flow'] = []
                merged['control_flow'].extend(slice_info['control_structures'])
            
            # Track maximum nesting depth
            slice_nesting = spec.get('nesting_depth', 0)
            if isinstance(slice_nesting, (int, float)):
                max_nesting_depth = max(max_nesting_depth, int(slice_nesting))
        
        # Build merged description
        if slice_descriptions:
            merged['description'] = f"Function decomposed into {len(slice_specs)} slices:\n" + '\n'.join(slice_descriptions)
        else:
            merged['description'] = f"Function decomposed into {len(slice_specs)} semantic slices"
        
        merged['user_stories'] = all_user_stories
        merged['test_cases'] = all_test_cases
        merged['preconditions'] = list(set(all_preconditions))
        merged['postconditions'] = list(set(all_postconditions))
        
        # Preserve variable names as list of dicts for better structure
        variable_names_list = []
        for var in sorted(all_variables):
            if isinstance(var, str) and var.strip():
                variable_names_list.append({
                    'name': var,
                    'purpose': 'Used across slices',
                    'preserve_exact_name': True
                })
        merged['variable_names'] = variable_names_list
        merged['variables'] = list(all_variables)  # Keep for backward compatibility
        
        # Deduplicate control flow
        if 'control_flow' in merged:
            merged['control_flow'] = list(dict.fromkeys(merged['control_flow']))  # Preserve order
        
        # Add metadata
        merged['specification_method'] = 'slice_by_slice'
        merged['num_slices'] = len(slice_specs)
        merged['max_nesting_depth'] = max_nesting_depth  # Include nesting depth in merged spec
        
        return merged

