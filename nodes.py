"""
Enhanced PocketFlow Nodes V2
Includes test generation, test execution, and behavioral validation
"""

import os
import ast
import json
import time
import re
import subprocess
import tempfile
import traceback
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from agents.advanced_analyzer import AdvancedCodeAnalyzer, SemanticSimilarityAnalyzer
from agents.smart_prompt_engine import SmartPromptEngine


class BaseNode:
    """Base class for all PocketFlow nodes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the node's processing"""
        raise NotImplementedError


class CodeAnalyzerNode(BaseNode):
    """Analyzes Python code structure and dependencies"""
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code structure and extract functions"""
        print("    Analyzing code structure...")
        
        project_path = context['project_path']
        
        python_files = self._find_python_files(project_path)
        
        if not python_files:
            raise ValueError("No Python files found in project")
        
        analyzed_files = {}
        all_functions = {}
        
        for file_path in python_files:
            print(f"      Analyzing {os.path.basename(file_path)}")
            
            try:
                file_analysis = self._analyze_file(file_path)
                analyzed_files[file_path] = file_analysis
                
                for func_name, func_info in file_analysis.get('functions', {}).items():
                    func_id = f"{file_path}::{func_name}"
                    all_functions[func_id] = {
                        'file_path': file_path,
                        'function_name': func_name,
                        'source_code': func_info['source'],
                        'complexity': func_info['complexity'],
                        'dependencies': func_info['calls'],
                        'line_number': func_info['line_number'],
                        'imports': file_analysis.get('imports', [])
                    }
            
            except Exception as e:
                print(f"        WARNING: Error analyzing {file_path}: {e}")
                continue
        
        context['analyzed_files'] = analyzed_files
        context['all_functions'] = all_functions
        context['total_functions'] = len(all_functions)
        
        print(f"    Found {len(all_functions)} functions across {len(analyzed_files)} files")
        
        return context
    
    def _find_python_files(self, project_path: str) -> List[str]:
        """Find all Python files in the project"""
        python_files = []
        
        for root, dirs, files in os.walk(project_path):
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', '.pytest_cache', 'node_modules', '.venv', 'venv']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        return python_files
    
    def _analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            analysis = {
                'file_path': file_path,
                'functions': self._extract_functions(tree, content),
                'imports': self._extract_imports(tree),
                'classes': self._extract_classes(tree, content),
                'complexity': self._calculate_file_complexity(tree)
            }
            
            return analysis
            
        except Exception as e:
            return {
                'file_path': file_path,
                'error': str(e),
                'functions': {},
                'imports': [],
                'classes': {},
                'complexity': 0
            }
    
    def _extract_functions(self, tree: ast.AST, content: str) -> Dict[str, Dict[str, Any]]:
        """Extract function information from AST"""
        functions = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    'name': node.name,
                    'line_number': node.lineno,
                    'args': [arg.arg for arg in node.args.args],
                    'defaults': [ast.unparse(default) for default in node.args.defaults],
                    'returns': ast.unparse(node.returns) if node.returns else None,
                    'decorators': [ast.unparse(dec) for dec in node.decorator_list],
                    'docstring': ast.get_docstring(node),
                    'complexity': self._calculate_function_complexity(node),
                    'calls': self._extract_function_calls(node),
                    'source': self._get_function_source(content, node)
                }
                functions[node.name] = func_info
        
        return functions
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")
        
        return imports
    
    def _extract_classes(self, tree: ast.AST, content: str) -> Dict[str, Dict[str, Any]]:
        """Extract class information"""
        classes = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'line_number': node.lineno,
                    'bases': [ast.unparse(base) for base in node.bases],
                    'decorators': [ast.unparse(dec) for dec in node.decorator_list],
                    'docstring': ast.get_docstring(node),
                    'methods': {}
                }
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_info = self._extract_functions(ast.Module([item]), content)
                        if method_info:
                            class_info['methods'][item.name] = list(method_info.values())[0]
                
                classes[node.name] = class_info
        
        return classes
    
    def _calculate_file_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of a file"""
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _extract_function_calls(self, node: ast.FunctionDef) -> List[str]:
        """Extract function calls from a function"""
        calls = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    if isinstance(child.func.value, ast.Name):
                        calls.append(f"{child.func.value.id}.{child.func.attr}")
        
        return calls
    
    def _get_function_source(self, content: str, node: ast.FunctionDef) -> str:
        """Extract source code for a function"""
        lines = content.splitlines()
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 1
        
        return '\n'.join(lines[start_line:end_line])


class SpecificationGeneratorNode(BaseNode):
    """Generates specifications for functions"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from utils.call_llm import call_llm
        self.call_llm = call_llm
        self.smart_prompt_engine = SmartPromptEngine()
        self.advanced_analyzer = AdvancedCodeAnalyzer()
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specifications for all functions"""
        print("    Generating specifications...")
        
        all_functions = context.get('all_functions', {})
        specifications = context.get('specifications', {})
        
        for func_id, func_info in all_functions.items():
            if func_id in specifications and specifications[func_id].get('success', False):
                continue
            
            print(f"      Processing {func_info['function_name']}...")
            
            try:
                spec_result = self._generate_specification(func_info, context)
                
                if spec_result['success']:
                    specifications[func_id] = {
                        'success': True,
                        'function_name': func_info['function_name'],
                        'file_path': func_info['file_path'],
                        'specification': spec_result['specification'],
                        'complexity': func_info['complexity'],
                        'dependencies': func_info['dependencies'],
                        'original_code': func_info['source_code'],
                        'imports': func_info.get('imports', [])
                    }
                    print(f"        Specification generated")
                else:
                    specifications[func_id] = {
                        'success': False,
                        'error': spec_result['error'],
                        'function_name': func_info['function_name'],
                        'file_path': func_info['file_path']
                    }
                    print(f"        ERROR: Failed: {spec_result['error']}")
            
            except Exception as e:
                specifications[func_id] = {
                    'success': False,
                    'error': str(e),
                    'function_name': func_info['function_name'],
                    'file_path': func_info['file_path']
                }
                print(f"        ERROR: {e}")
        
        context['specifications'] = specifications
        print(f"    Generated {len([s for s in specifications.values() if s.get('success', False)])} specifications")
        
        return context
    
    def _generate_specification(self, func_info: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specification for a single function"""
        source_code = func_info['source_code']
        complexity = func_info['complexity']
        
        code_analysis = self.advanced_analyzer.analyze_code_advanced(source_code)
        
        # Check if this is a class method
        is_class_method = 'self' in func_info.get('source_code', '')
        class_context = self._extract_class_context(func_info, context) if is_class_method else None
        
        prompt_context = {
            'complexity': complexity,
            'dependencies': func_info.get('dependencies', []),
            'file_path': func_info.get('file_path', ''),
            'function_name': func_info.get('function_name', ''),
            'code_analysis': code_analysis,
            'imports': func_info.get('imports', []),
            'is_class_method': is_class_method,
            'class_context': class_context
        }
        
        feedback = context.get('feedback_data', {}).get(f"{func_info['file_path']}::{func_info['function_name']}", None)
        similarity_gaps = []
        iteration = context.get('current_iteration', 1)
        
        if feedback:
            similarity_gaps = feedback.get('gaps', [])
        
        prompt = self.smart_prompt_engine.generate_adaptive_prompt(
            source_code, prompt_context, similarity_gaps, iteration
        )
        
        try:
            response = self.call_llm(prompt)
            specification = self._parse_specification_response(response)
            
            # Add class context to specification if applicable
            if is_class_method and class_context:
                specification['class_context'] = class_context
            
            return {
                'success': True,
                'specification': specification,
                'code_analysis': code_analysis
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_class_context(self, func_info: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract class context for class methods"""
        try:
            file_path = func_info.get('file_path', '')
            if file_path not in context.get('analyzed_files', {}):
                return {}
            
            file_analysis = context['analyzed_files'][file_path]
            classes = file_analysis.get('classes', {})
            
            # Find which class this method belongs to
            for class_name, class_info in classes.items():
                if func_info['function_name'] in class_info.get('methods', {}):
                    return {
                        'class_name': class_name,
                        'class_docstring': class_info.get('docstring', ''),
                        'class_bases': class_info.get('bases', []),
                        'other_methods': list(class_info.get('methods', {}).keys()),
                        'class_attributes': self._extract_class_attributes(class_info)
                    }
            
            return {}
        except Exception:
            return {}
    
    def _extract_class_attributes(self, class_info: Dict[str, Any]) -> List[str]:
        """Extract class attributes from __init__ method"""
        try:
            init_method = class_info.get('methods', {}).get('__init__', {})
            if not init_method:
                return []
            
            source = init_method.get('source', '')
            attributes = []
            
            # Find self.attribute assignments
            import re
            pattern = r'self\.(\w+)\s*='
            matches = re.findall(pattern, source)
            attributes.extend(matches)
            
            return list(set(attributes))
        except Exception:
            return []
    
    def _parse_specification_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured specification"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    'raw_specification': response,
                    'parsed': False
                }
        except json.JSONDecodeError:
            return {
                'raw_specification': response,
                'parsed': False
            }


class CodeRegenerationNode(BaseNode):
    """Regenerates code from specifications"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from utils.call_llm import call_llm
        self.call_llm = call_llm
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Regenerate code for all functions"""
        print("    Regenerating code from specifications...")
        
        specifications = context.get('specifications', {})
        regenerated_code = context.get('regenerated_code', {})
        
        for func_id, spec_data in specifications.items():
            if not spec_data.get('success', False):
                continue
            
            if func_id in regenerated_code:
                continue
            
            print(f"      Regenerating {spec_data['function_name']}...")
            
            try:
                code = self._regenerate_code(spec_data['specification'])
                
                if code:
                    regenerated_code[func_id] = {
                        'code': code,
                        'function_name': spec_data['function_name'],
                        'file_path': spec_data['file_path']
                    }
                    print(f"        Code regenerated")
                else:
                    print(f"        ERROR: Failed to regenerate code")
            
            except Exception as e:
                print(f"        ERROR: {e}")
                continue
        
        context['regenerated_code'] = regenerated_code
        print(f"    Regenerated {len(regenerated_code)} functions")
        
        return context
    
    def _regenerate_code(self, specification: Dict[str, Any]) -> Optional[str]:
        """Regenerate code from specification"""
        
        # Check if this is a class method
        is_class_method = 'class_context' in specification
        
        if is_class_method:
            class_ctx = specification['class_context']
            prompt = f"""
Generate ONLY the method code (not the entire class) based on this specification.

Specification:
{json.dumps(specification, indent=2)}

IMPORTANT: 
- Generate ONLY the method definition, NOT the entire class
- Include 'self' as first parameter
- Use the class attributes: {class_ctx.get('class_attributes', [])}
- The method belongs to class: {class_ctx.get('class_name', 'Unknown')}

Generate only the method code starting with 'def method_name(self, ...):' 
Do NOT include the class definition, do NOT include __init__, just the method itself.
No explanations or markdown formatting.
"""
        else:
            prompt = f"""
Generate Python code based on the following detailed specification. The code should match the specification exactly.

Specification:
{json.dumps(specification, indent=2)}

Generate only the Python function code, no explanations or markdown formatting.
The code should be complete and runnable.
"""
        
        try:
            response = self.call_llm(prompt)
            code = self._clean_generated_code(response)
            
            # If this is a class method and we got a full class, extract just the method
            if is_class_method and 'class ' in code.lower():
                code = self._extract_method_from_class(code, specification.get('function_name', ''))
            
            return code
        except Exception:
            return None
    
    def _extract_method_from_class(self, code: str, method_name: str) -> str:
        """Extract just the method from a full class definition"""
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == method_name:
                            # Extract this method's source
                            lines = code.splitlines()
                            start = item.lineno - 1
                            end = item.end_lineno if hasattr(item, 'end_lineno') else start + 10
                            return '\n'.join(lines[start:end])
            
            # If extraction failed, return original (might already be just method)
            return code
        except:
            return code
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean generated code to remove common artifacts"""
        code = re.sub(r'```python\s*\n?', '', code)
        code = re.sub(r'```\s*$', '', code)
        code = re.sub(r'```\s*\n?', '', code)
        code = re.sub(r'^Here is the code:\s*\n?', '', code, flags=re.MULTILINE)
        code = re.sub(r'^Here\'s the code:\s*\n?', '', code, flags=re.MULTILINE)
        code = re.sub(r'^The code is:\s*\n?', '', code, flags=re.MULTILINE)
        
        return code.strip()


class TestGenerationNode(BaseNode):
    """Generates tests for behavioral equivalence validation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from utils.call_llm import call_llm
        self.call_llm = call_llm
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate tests for all functions"""
        print("    Generating tests for behavioral validation...")
        
        specifications = context.get('specifications', {})
        generated_tests = context.get('generated_tests', {})
        
        for func_id, spec_data in specifications.items():
            if not spec_data.get('success', False):
                continue
            
            if func_id in generated_tests:
                continue
            
            print(f"      Generating tests for {spec_data['function_name']}...")
            
            try:
                tests = self._generate_tests(
                    spec_data['original_code'],
                    spec_data['specification'],
                    spec_data['function_name']
                )
                
                if tests:
                    generated_tests[func_id] = {
                        'tests': tests,
                        'function_name': spec_data['function_name'],
                        'file_path': spec_data['file_path']
                    }
                    print(f"        Generated {len(tests)} tests")
                else:
                    print(f"        ERROR: Failed to generate tests")
            
            except Exception as e:
                print(f"        ERROR: {e}")
                continue
        
        context['generated_tests'] = generated_tests
        print(f"    Generated tests for {len(generated_tests)} functions")
        
        return context
    
    def _generate_tests(self, original_code: str, specification: Dict[str, Any], function_name: str) -> List[Dict[str, Any]]:
        """Generate tests for a function"""
        prompt = f"""
Generate comprehensive unit tests for the following Python function.

Original Function:
```python
{original_code}
```

Specification:
{json.dumps(specification, indent=2)}

Generate test cases that:
1. Test normal behavior with typical inputs
2. Test edge cases and boundary conditions
3. Test error handling and exceptions
4. Test with various data types if applicable
5. Test all documented behavior

Format: Return a JSON array of test cases with this structure:
[
  {{
    "test_name": "test_description",
    "inputs": {{"arg1": value1, "arg2": value2}},
    "expected_output": expected_value,
    "expected_exception": "ExceptionName" or null,
    "description": "What this test validates"
  }}
]

Generate at least 5-10 test cases covering all aspects of the function.
Return ONLY the JSON array, no other text.
"""
        
        try:
            response = self.call_llm(prompt)
            
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                tests = json.loads(json_match.group())
                return tests
            else:
                return []
                
        except Exception:
            return []


class TestExecutionNode(BaseNode):
    """Executes tests on both original and regenerated code"""
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tests on all functions"""
        print("    Executing tests on original and regenerated code...")
        
        specifications = context.get('specifications', {})
        regenerated_code = context.get('regenerated_code', {})
        generated_tests = context.get('generated_tests', {})
        test_results = context.get('test_results', {})
        
        for func_id in specifications.keys():
            if not specifications[func_id].get('success', False):
                continue
            
            if func_id not in regenerated_code or func_id not in generated_tests:
                continue
            
            if func_id in test_results:
                continue
            
            print(f"      Testing {specifications[func_id]['function_name']}...")
            
            try:
                results = self._execute_tests(
                    specifications[func_id]['original_code'],
                    regenerated_code[func_id]['code'],
                    generated_tests[func_id]['tests'],
                    specifications[func_id]['function_name'],
                    specifications[func_id].get('imports', [])
                )
                
                test_results[func_id] = results
                
                passed = results['original_passed'] + results['regenerated_passed']
                total = results['total_tests'] * 2
                print(f"        Tests passed: {passed}/{total}")
                
            except Exception as e:
                print(f"        ERROR: {e}")
                continue
        
        context['test_results'] = test_results
        print(f"    Executed tests for {len(test_results)} functions")
        
        return context
    
    def _execute_tests(self, original_code: str, regenerated_code: str, tests: List[Dict[str, Any]], 
                       function_name: str, imports: List[str]) -> Dict[str, Any]:
        """Execute tests on both code versions"""
        
        results = {
            'total_tests': len(tests),
            'original_passed': 0,
            'original_failed': 0,
            'regenerated_passed': 0,
            'regenerated_failed': 0,
            'failures': [],
            'behavioral_match': True
        }
        
        for test in tests:
            original_result = self._run_single_test(original_code, test, function_name, imports)
            regenerated_result = self._run_single_test(regenerated_code, test, function_name, imports)
            
            if original_result['passed']:
                results['original_passed'] += 1
            else:
                results['original_failed'] += 1
            
            if regenerated_result['passed']:
                results['regenerated_passed'] += 1
            else:
                results['regenerated_failed'] += 1
            
            if original_result['output'] != regenerated_result['output']:
                results['behavioral_match'] = False
                results['failures'].append({
                    'test': test,
                    'original_output': original_result['output'],
                    'regenerated_output': regenerated_result['output'],
                    'original_error': original_result.get('error'),
                    'regenerated_error': regenerated_result.get('error')
                })
        
        return results
    
    def _run_single_test(self, code: str, test: Dict[str, Any], function_name: str, imports: List[str]) -> Dict[str, Any]:
        """Run a single test case"""
        try:
            import_lines = '\n'.join([f"import {imp.split('.')[0]}" for imp in imports if imp])
            
            test_code = f"""
{import_lines}

{code}

result = {function_name}(**{test['inputs']})
"""
            
            local_vars = {}
            exec(test_code, {}, local_vars)
            
            result = local_vars.get('result')
            expected = test.get('expected_output')
            
            passed = result == expected
            
            return {
                'passed': passed,
                'output': result,
                'expected': expected
            }
            
        except Exception as e:
            expected_exception = test.get('expected_exception')
            
            if expected_exception and type(e).__name__ == expected_exception:
                return {
                    'passed': True,
                    'output': f"Exception: {type(e).__name__}",
                    'expected': f"Exception: {expected_exception}"
                }
            else:
                return {
                    'passed': False,
                    'output': None,
                    'expected': test.get('expected_output'),
                    'error': str(e)
                }


class SimilarityAnalyzerNode(BaseNode):
    """Analyzes similarity between original and regenerated code"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.semantic_analyzer = SemanticSimilarityAnalyzer()
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze similarity for all functions"""
        print("    Analyzing similarity...")
        
        specifications = context.get('specifications', {})
        regenerated_code = context.get('regenerated_code', {})
        test_results = context.get('test_results', {})
        similarity_results = context.get('similarity_results', {})
        
        for func_id in specifications.keys():
            if not specifications[func_id].get('success', False):
                continue
            
            if func_id not in regenerated_code:
                continue
            
            print(f"      Analyzing {specifications[func_id]['function_name']}...")
            
            try:
                original_code = specifications[func_id]['original_code']
                regen_code = regenerated_code[func_id]['code']
                
                similarity_metrics = self.semantic_analyzer.calculate_semantic_similarity(
                    original_code, regen_code
                )
                
                if func_id in test_results:
                    test_data = test_results[func_id]
                    behavioral_similarity = self._calculate_behavioral_similarity(test_data)
                    similarity_metrics['behavioral_test_similarity'] = behavioral_similarity
                
                overall_similarity = self._calculate_enhanced_overall_similarity(similarity_metrics)
                similarity_metrics['overall_similarity'] = overall_similarity
                
                similarity_results[func_id] = {
                    'original_code': original_code,
                    'regenerated_code': regen_code,
                    'similarity_metrics': similarity_metrics,
                    'overall_similarity': overall_similarity,
                    'test_based_validation': func_id in test_results
                }
                
                print(f"        Similarity: {overall_similarity:.1%}")
                
            except Exception as e:
                print(f"        ERROR: {e}")
                continue
        
        context['similarity_results'] = similarity_results
        
        similarities = [r['overall_similarity'] for r in similarity_results.values()]
        if 'similarity_history' not in context:
            context['similarity_history'] = []
        context['similarity_history'].extend(similarities)
        
        print(f"    Analyzed {len(similarity_results)} functions")
        
        return context
    
    def _calculate_behavioral_similarity(self, test_results: Dict[str, Any]) -> float:
        """Calculate behavioral similarity from test results"""
        if not test_results.get('behavioral_match', False):
            total_tests = test_results['total_tests']
            matching_tests = total_tests - len(test_results['failures'])
            return matching_tests / total_tests if total_tests > 0 else 0.0
        else:
            return 1.0
    
    def _calculate_enhanced_overall_similarity(self, metrics: Dict[str, float]) -> float:
        """
        Calculate enhanced overall similarity with test-based validation
        FOCUS: Structural correctness and behavioral equivalence matter most
        Variable names, formatting, and exact text don't matter if behavior is correct
        """
        weights = {
            'textual_similarity': 0.05,  # Reduced - exact text doesn't matter
            'structural_similarity': 0.35,  # Increased - AST structure matters
            'behavioral_similarity': 0.25,  # Important - behavior patterns
            'semantic_similarity': 0.10,  # Reduced - meaning captured by structure
            'behavioral_test_similarity': 0.25  # Increased - runtime correctness is key
        }
        
        weighted_sum = sum(metrics.get(metric, 0) * weight for metric, weight in weights.items())
        return weighted_sum


class FeedbackLoopNode(BaseNode):
    """First feedback loop: Modifies prompt based on similarity gaps"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.smart_prompt_engine = SmartPromptEngine()
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process first feedback loop: modify prompts"""
        print("    Processing feedback loop (prompt modification)...")
        
        similarity_results = context.get('similarity_results', {})
        target_similarity = context['target_similarity']
        
        if 'feedback_data' not in context:
            context['feedback_data'] = {}
        
        improved_count = 0
        
        for func_id, result in similarity_results.items():
            if result['overall_similarity'] < target_similarity:
                print(f"      Analyzing gaps for {func_id}...")
                
                try:
                    gaps = self._analyze_similarity_gaps(result)
                    
                    context['feedback_data'][func_id] = {
                        'gaps': gaps,
                        'metrics': result['similarity_metrics'],
                        'iteration': context.get('current_iteration', 1)
                    }
                    
                    improved_count += 1
                    print(f"        Feedback prepared")
                
                except Exception as e:
                    print(f"        ERROR: {e}")
                    continue
        
        print(f"    Prepared feedback for {improved_count} functions")
        
        return context
    
    def _analyze_similarity_gaps(self, result: Dict[str, Any]) -> List[str]:
        """Analyze gaps in similarity"""
        gaps = []
        metrics = result['similarity_metrics']
        
        if metrics.get('structural_similarity', 0) < 0.8:
            gaps.append("Structural differences: Code organization and AST structure differ")
        
        if metrics.get('semantic_similarity', 0) < 0.8:
            gaps.append("Semantic differences: Code meaning and context differ")
        
        if metrics.get('behavioral_similarity', 0) < 0.8:
            gaps.append("Behavioral differences: Function behavior patterns differ")
        
        if metrics.get('textual_similarity', 0) < 0.8:
            gaps.append("Textual differences: Surface-level code text differs")
        
        if metrics.get('behavioral_test_similarity', 0) < 0.8:
            gaps.append("Test-based validation: Functions produce different outputs for same inputs")
        
        return gaps


class RuntimeFeedbackLoopNode(BaseNode):
    """Second feedback loop: Appends test failures without modifying prompt"""
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process second feedback loop: append test failures"""
        print("    Processing runtime feedback loop (test failures)...")
        
        test_results = context.get('test_results', {})
        specifications = context.get('specifications', {})
        
        if 'runtime_feedback' not in context:
            context['runtime_feedback'] = {}
        
        feedback_count = 0
        
        for func_id, results in test_results.items():
            if not results.get('behavioral_match', True):
                print(f"      Recording test failures for {func_id}...")
                
                try:
                    failure_summary = self._summarize_test_failures(results)
                    
                    if func_id not in context['runtime_feedback']:
                        context['runtime_feedback'][func_id] = []
                    
                    context['runtime_feedback'][func_id].append({
                        'iteration': context.get('current_iteration', 1),
                        'failures': failure_summary,
                        'total_failures': len(results['failures'])
                    })
                    
                    if func_id in specifications:
                        if 'appended_failures' not in specifications[func_id]['specification']:
                            specifications[func_id]['specification']['appended_failures'] = []
                        
                        specifications[func_id]['specification']['appended_failures'].extend(
                            results['failures']
                        )
                    
                    feedback_count += 1
                    print(f"        Recorded {len(results['failures'])} failures")
                
                except Exception as e:
                    print(f"        ERROR: {e}")
                    continue
        
        print(f"    Recorded runtime feedback for {feedback_count} functions")
        
        return context
    
    def _summarize_test_failures(self, test_results: Dict[str, Any]) -> List[str]:
        """Summarize test failures"""
        summaries = []
        
        for failure in test_results['failures']:
            test_name = failure['test'].get('test_name', 'Unknown test')
            original_out = failure.get('original_output', 'N/A')
            regen_out = failure.get('regenerated_output', 'N/A')
            
            summary = f"Test '{test_name}': Original output: {original_out}, Regenerated output: {regen_out}"
            summaries.append(summary)
        
        return summaries


class ConvergenceCheckerNode(BaseNode):
    """Checks for convergence in the iterative process"""
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if the process has converged"""
        print("    Checking convergence...")
        
        similarity_results = context.get('similarity_results', {})
        target_similarity = context['target_similarity']
        current_iteration = context['current_iteration']
        max_iterations = context['max_iterations']
        
        target_achieved_count = sum(
            1 for result in similarity_results.values()
            if result['overall_similarity'] >= target_similarity
        )
        
        total_functions = len(similarity_results)
        convergence_rate = target_achieved_count / total_functions if total_functions > 0 else 0
        
        converged = False
        reason = ""
        
        if convergence_rate >= 0.8:
            converged = True
            reason = f"Target similarity achieved for {target_achieved_count}/{total_functions} functions"
        elif current_iteration >= max_iterations:
            converged = True
            reason = f"Maximum iterations ({max_iterations}) reached"
        elif current_iteration >= 3:
            recent_similarities = context.get('similarity_history', [])[-total_functions:]
            if len(recent_similarities) >= total_functions:
                prev_similarities = context.get('similarity_history', [])[-2*total_functions:-total_functions]
                if len(prev_similarities) >= total_functions:
                    avg_improvement = np.mean(recent_similarities) - np.mean(prev_similarities)
                    if avg_improvement < 0.01:
                        converged = True
                        reason = "No significant improvement in recent iterations"
        
        context['convergence_achieved'] = converged
        context['convergence_reason'] = reason
        context['convergence_rate'] = convergence_rate
        
        if converged:
            print(f"    Convergence achieved: {reason}")
        else:
            print(f"    Continuing iteration (convergence rate: {convergence_rate:.1%})")
        
        return context

