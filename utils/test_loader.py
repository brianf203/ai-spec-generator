"""
Utility to load user-provided test cases from unittest test files
"""
import ast
import os
import re
from typing import Dict, List, Any, Optional
from pathlib import Path


class TestLoader:
    """Loads test cases from unittest test files"""
    
    @staticmethod
    def find_test_file(source_file_path: str, project_path: str) -> Optional[str]:
        """
        Find the corresponding test file for a source file.
        Looks for test_*.py in: same dir, tests/ subdir, or test_projects/{size}/
        """
        source_path = Path(source_file_path)
        source_name = source_path.stem  # e.g., 'operations', 'simple_calc'
        project_path_obj = Path(project_path).resolve()

        # Same directory: test_{name}.py
        test_file = source_path.parent / f"test_{source_name}.py"
        if test_file.exists():
            return str(test_file)

        # tests/ subdirectory (common in real projects)
        for candidate in [source_path.parent / "tests", project_path_obj / "tests"]:
            tf = candidate / f"test_{source_name}.py"
            if tf.exists():
                return str(tf)

        # test_projects structure: test_projects/{size}/test_{name}.py
        for size_dir in ['small', 'medium', 'large']:
            test_file = project_path_obj / 'test_projects' / size_dir / f"test_{source_name}.py"
            if test_file.exists():
                return str(test_file)

        # Any test_*.py in same directory
        for test_file in source_path.parent.glob("test_*.py"):
            return str(test_file)

        return None
    
    @staticmethod
    def parse_unittest_file(test_file_path: str, function_name: str) -> List[Dict[str, Any]]:
        """
        Parse a unittest test file and extract test cases for a specific function.
        Returns a list of test case dictionaries matching the expected format.
        """
        if not os.path.exists(test_file_path):
            return []
        
        try:
            with open(test_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"      ERROR: Failed to read test file {test_file_path}: {e}")
            return []
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"      ERROR: Failed to parse test file {test_file_path}: {e}")
            return []
        
        test_cases = []
        
        # Find all test classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if it's a test class (inherits from unittest.TestCase or has test methods)
                is_test_class = False
                for base in node.bases:
                    # Check for unittest.TestCase
                    if isinstance(base, ast.Attribute):
                        if isinstance(base.value, ast.Name) and base.value.id == 'unittest' and base.attr == 'TestCase':
                            is_test_class = True
                            break
                    # Check for direct TestCase (if imported)
                    elif isinstance(base, ast.Name) and base.id == 'TestCase':
                        is_test_class = True
                        break
                
                # Also check if class has test methods (fallback)
                if not is_test_class:
                    has_test_methods = any(
                        isinstance(item, ast.FunctionDef) and item.name.startswith('test_')
                        for item in node.body
                    )
                    if has_test_methods:
                        is_test_class = True
                
                if is_test_class:
                    # Extract test methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name.startswith('test_'):
                            result = TestLoader._extract_test_case(item, function_name, content)
                            if result:
                                # Result can be a single test case or a list
                                if isinstance(result, list):
                                    test_cases.extend(result)
                                else:
                                    test_cases.append(result)
        
        return test_cases
    
    @staticmethod
    def _extract_test_case(test_method: ast.FunctionDef, function_name: str, file_content: str) -> Optional[Dict[str, Any]]:
        """Extract a single test case from a test method"""
        test_name = test_method.name
        
        # Extract all test cases from this method (may have multiple assertions)
        test_cases = []
        
        # Look for assertEqual/assertRaises statements that test our function
        for stmt in test_method.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if isinstance(call.func, ast.Attribute):
                    # Check for assertEqual(function_call, expected)
                    if call.func.attr == 'assertEqual' and len(call.args) >= 2:
                        # First arg should be the function call
                        func_call = call.args[0]
                        if isinstance(func_call, ast.Call) and isinstance(func_call.func, ast.Name):
                            if func_call.func.id == function_name:
                                # Extract arguments from function call
                                args = []
                                for arg in func_call.args:
                                    if isinstance(arg, ast.Constant):
                                        args.append(arg.value)
                                    elif isinstance(arg, ast.UnaryOp) and isinstance(arg.operand, ast.Constant):
                                        # Handle negative numbers: -1
                                        if isinstance(arg.op, ast.USub):
                                            args.append(-arg.operand.value)
                                        else:
                                            args.append(arg.operand.value)
                                
                                # Extract expected output
                                expected_output = None
                                if isinstance(call.args[1], ast.Constant):
                                    expected_output = call.args[1].value
                                
                                # Build inputs dict
                                inputs = {}
                                if len(args) == 1:
                                    inputs = {"arg": args[0]}
                                elif len(args) == 2:
                                    inputs = {"a": args[0], "b": args[1]}
                                else:
                                    inputs = {f"arg{i}": args[i] for i in range(len(args))}
                                
                                if inputs or expected_output is not None:
                                    test_cases.append({
                                        "test_name": f"{test_name}_{len(test_cases)}",
                                        "inputs": inputs,
                                        "expected_output": expected_output,
                                        "expected_exception": None,
                                        "description": f"Test case from {test_name}",
                                        "state_assertions": []
                                    })
                    
                    # Check for assertRaises(Exception, function_call, ...)
                    elif call.func.attr == 'assertRaises' and len(call.args) >= 2:
                        exception_type = call.args[0]
                        func_call = call.args[1]
                        if isinstance(func_call, ast.Call) and isinstance(func_call.func, ast.Name):
                            if func_call.func.id == function_name:
                                # Extract arguments
                                args = []
                                for arg in func_call.args:
                                    if isinstance(arg, ast.Constant):
                                        args.append(arg.value)
                                
                                # Extract exception type
                                expected_exception = None
                                if isinstance(exception_type, ast.Name):
                                    expected_exception = exception_type.id
                                
                                # Build inputs dict
                                inputs = {}
                                if len(args) == 1:
                                    inputs = {"arg": args[0]}
                                elif len(args) == 2:
                                    inputs = {"a": args[0], "b": args[1]}
                                else:
                                    inputs = {f"arg{i}": args[i] for i in range(len(args))}
                                
                                if inputs or expected_exception:
                                    test_cases.append({
                                        "test_name": f"{test_name}_{len(test_cases)}",
                                        "inputs": inputs,
                                        "expected_output": None,
                                        "expected_exception": expected_exception,
                                        "description": f"Test case from {test_name}",
                                "state_assertions": []
                            })
        
        # Return test cases if found
        return test_cases if test_cases else None
        
        # If no structured extraction worked, try regex-based extraction (fallback)
        # This code is unreachable but kept for reference
        if False and not test_cases:
            method_source = None
            if hasattr(ast, 'get_source_segment'):
                try:
                    method_source = ast.get_source_segment(file_content, test_method)
                except:
                    pass
            
            if not method_source:
                # Fallback: extract method source manually
                try:
                    start_line = test_method.lineno - 1
                    end_line = test_method.end_lineno if hasattr(test_method, 'end_lineno') else start_line + 20
                    lines = file_content.split('\n')
                    method_source = '\n'.join(lines[start_line:end_line])
                except:
                    method_source = None
            
            if method_source:
                # Find all function calls
                call_pattern = rf'{re.escape(function_name)}\s*\(([^)]*)\)'
                calls = re.finditer(call_pattern, method_source)
                
                for call_match in calls:
                    args_str = call_match.group(1)
                    inputs = {}
                    
                    # Parse arguments
                    if args_str.strip():
                        try:
                            # Try to evaluate as Python literal
                            if ',' in args_str:
                                parts = [p.strip() for p in args_str.split(',')]
                                if len(parts) == 2:
                                    try:
                                        inputs = {
                                            "a": eval(parts[0]),
                                            "b": eval(parts[1])
                                        }
                                    except:
                                        pass
                            else:
                                try:
                                    inputs = {"arg": eval(args_str)}
                                except:
                                    pass
                        except:
                            pass
                    
                    # Find corresponding assertEqual
                    assert_pattern = r'assertEqual\s*\([^,]+,\s*([^)]+)\)'
                    assert_match = re.search(assert_pattern, method_source)
                    expected_output = None
                    if assert_match:
                        try:
                            expected_output = eval(assert_match.group(1))
                        except:
                            pass
                    
                    # Find assertRaises
                    raises_pattern = rf'assertRaises\s*\(\s*(\w+)\s*,\s*[^,]*{re.escape(function_name)}'
                    raises_match = re.search(raises_pattern, method_source)
                    expected_exception = None
                    if raises_match:
                        expected_exception = raises_match.group(1)
                    
                    if inputs or expected_output is not None or expected_exception:
                        test_cases.append({
                            "test_name": f"{test_name}_{len(test_cases)}",
                            "inputs": inputs,
                            "expected_output": expected_output,
                            "expected_exception": expected_exception,
                            "description": f"Test case from {test_name}",
                            "state_assertions": []
                        })
        
        # Return all test cases found in this method
        return test_cases if test_cases else None
    
    @staticmethod
    def load_tests_for_function(source_file_path: str, function_name: str, project_path: str) -> List[Dict[str, Any]]:
        """
        Main entry point: Load tests for a specific function from its test file.
        """
        test_file = TestLoader.find_test_file(source_file_path, project_path)
        if not test_file:
            return []
        
        return TestLoader.parse_unittest_file(test_file, function_name)
