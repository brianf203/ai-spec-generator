"""
Example-Driven Specification Enhancement
Uses concrete examples from test cases and runtime behavior to enhance specifications
"""

from typing import Dict, List, Any, Optional, Set
import ast
import json
from collections import defaultdict


class ExampleDrivenSpecEnhancer:
    """Enhances specifications with concrete examples derived from code analysis"""
    
    def __init__(self):
        self.example_library: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def extract_examples_from_code(
        self,
        code: str,
        function_name: str
    ) -> Dict[str, List[Any]]:
        """Extract concrete examples from code by analyzing patterns"""
        examples = {
            'input_examples': [],
            'output_examples': [],
            'edge_case_examples': [],
            'exception_examples': []
        }
        
        try:
            tree = ast.parse(code)
            
            # Find literals that might be examples
            for node in ast.walk(tree):
                # Look for return statements with literals
                if isinstance(node, ast.Return) and node.value:
                    example_value = self._extract_literal_value(node.value)
                    if example_value is not None:
                        examples['output_examples'].append(example_value)
                
                # Look for comparisons with literals (edge cases)
                if isinstance(node, ast.Compare):
                    for comparator in node.comparators:
                        literal = self._extract_literal_value(comparator)
                        if literal is not None:
                            examples['edge_case_examples'].append({
                                'condition': ast.unparse(node) if hasattr(ast, 'unparse') else str(node),
                                'value': literal
                            })
                
                # Look for if conditions checking for None, empty, etc.
                if isinstance(node, ast.If):
                    condition = node.test
                    if isinstance(condition, ast.Compare):
                        for op in condition.ops:
                            if isinstance(op, ast.Is) or isinstance(op, ast.IsNot):
                                examples['edge_case_examples'].append({
                                    'type': 'None_check',
                                    'condition': ast.unparse(condition) if hasattr(ast, 'unparse') else str(condition)
                                })
            
            # Look for function calls that might indicate usage patterns
            call_examples = self._extract_call_patterns(tree, function_name)
            examples['input_examples'].extend(call_examples)
        
        except Exception:
            pass
        
        return examples
    
    def extract_examples_from_tests(
        self,
        test_results: Dict[str, Any]
    ) -> Dict[str, List[Any]]:
        """Extract examples from test execution results"""
        examples = {
            'working_inputs': [],
            'working_outputs': [],
            'failure_inputs': [],
            'exception_cases': []
        }
        
        # Extract from passing tests
        passes = test_results.get('passes', [])
        for test_pass in passes:
            test = test_pass.get('test', {})
            inputs = test.get('inputs', {})
            output = test_pass.get('original_output')
            
            if inputs:
                examples['working_inputs'].append(inputs)
            if output is not None:
                examples['working_outputs'].append(output)
        
        # Extract from failing tests
        failures = test_results.get('failures', [])
        for failure in failures:
            test = failure.get('test', {})
            inputs = test.get('inputs', {})
            original_exc = failure.get('original_exception')
            regen_exc = failure.get('regenerated_exception')
            
            if inputs:
                if original_exc or regen_exc:
                    examples['exception_cases'].append({
                        'inputs': inputs,
                        'exception': original_exc or regen_exc
                    })
                else:
                    examples['failure_inputs'].append(inputs)
        
        return examples
    
    def enhance_specification_with_examples(
        self,
        specification: Dict[str, Any],
        code_examples: Dict[str, List[Any]],
        test_examples: Optional[Dict[str, List[Any]]] = None
    ) -> Dict[str, Any]:
        """Enhance specification with concrete examples"""
        enhanced_spec = specification.copy()
        
        # Add code-derived examples
        if code_examples.get('output_examples'):
            if 'examples' not in enhanced_spec:
                enhanced_spec['examples'] = {}
            enhanced_spec['examples']['output_patterns'] = code_examples['output_examples'][:5]
        
        if code_examples.get('edge_case_examples'):
            edge_cases = enhanced_spec.get('edge_cases', [])
            for edge_example in code_examples['edge_case_examples'][:3]:
                edge_cases.append({
                    'source': 'code_analysis',
                    'details': edge_example
                })
            enhanced_spec['edge_cases'] = edge_cases
        
        # Add test-derived examples
        if test_examples:
            if 'examples' not in enhanced_spec:
                enhanced_spec['examples'] = {}
            
            if test_examples.get('working_inputs'):
                enhanced_spec['examples']['working_inputs'] = test_examples['working_inputs'][:5]
            
            if test_examples.get('working_outputs'):
                enhanced_spec['examples']['working_outputs'] = test_examples['working_outputs'][:5]
            
            if test_examples.get('exception_cases'):
                enhanced_spec['examples']['exception_cases'] = test_examples['exception_cases'][:3]
            
            # Enhance test matrix with examples
            test_matrix = enhanced_spec.get('test_matrix', [])
            if not test_matrix and test_examples.get('working_inputs'):
                for i, inputs in enumerate(test_examples['working_inputs'][:3]):
                    test_matrix.append({
                        'name': f'example_test_{i+1}',
                        'inputs': inputs,
                        'source': 'extracted_from_tests'
                    })
                enhanced_spec['test_matrix'] = test_matrix
        
        return enhanced_spec
    
    def generate_example_based_prompt_enhancement(
        self,
        specification: Dict[str, Any]
    ) -> str:
        """Generate prompt enhancement section based on examples in spec"""
        examples = specification.get('examples', {})
        if not examples:
            return ""
        
        enhancement = "\n\nCONCRETE EXAMPLES FOR REFERENCE:\n"
        
        if examples.get('working_inputs'):
            enhancement += "Working input examples:\n"
            for i, inputs in enumerate(examples['working_inputs'][:3], 1):
                enhancement += f"  {i}. {json.dumps(inputs)}\n"
        
        if examples.get('working_outputs'):
            enhancement += "\nWorking output examples:\n"
            for i, output in enumerate(examples['working_outputs'][:3], 1):
                enhancement += f"  {i}. {json.dumps(output)}\n"
        
        if examples.get('exception_cases'):
            enhancement += "\nException handling examples:\n"
            for i, exc_case in enumerate(examples['exception_cases'][:2], 1):
                enhancement += f"  {i}. Input: {json.dumps(exc_case.get('inputs', {}))} raises: {exc_case.get('exception')}\n"
        
        enhancement += "\nUse these examples as concrete reference points. Your regenerated code must handle these exact cases.\n"
        
        return enhancement
    
    def _extract_literal_value(self, node: ast.AST) -> Any:
        """Extract literal value from AST node"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.Str):  # Python < 3.8
            return node.s
        elif isinstance(node, ast.NameConstant):  # Python < 3.8
            return node.value
        elif isinstance(node, (ast.List, ast.Tuple)):
            return [self._extract_literal_value(item) for item in node.elts]
        elif isinstance(node, ast.Dict):
            return {
                self._extract_literal_value(k): self._extract_literal_value(v)
                for k, v in zip(node.keys, node.values)
            }
        return None
    
    def _extract_call_patterns(
        self,
        tree: ast.AST,
        function_name: str
    ) -> List[Dict[str, Any]]:
        """Extract function call patterns from AST"""
        patterns = []
        
        # Look for calls to the function itself (recursive patterns)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == function_name:
                    # Extract arguments
                    args = []
                    for arg in node.args:
                        literal = self._extract_literal_value(arg)
                        args.append(literal if literal is not None else 'variable')
                    patterns.append({'type': 'recursive_call', 'args': args})
        
        return patterns

