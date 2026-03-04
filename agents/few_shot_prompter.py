"""
Advanced Few-shot prompting with adaptive example selection
Supports code generation, specification generation, and test generation
Based on research showing that examples help LLMs preserve structure and improve accuracy
"""

from typing import Dict, Any, List, Optional
import ast
import re


class FewShotPromptEnhancer:
    """Enhances prompts with few-shot examples to improve code generation, spec generation, and test generation"""
    
    def __init__(self):
        self.code_examples = self._load_code_examples()
        self.spec_examples = self._load_spec_examples()
        self.test_examples = self._load_test_examples()
    
    def _load_code_examples(self) -> List[Dict[str, Any]]:
        """Load few-shot examples demonstrating good code regeneration"""
        return [
            {
                "specification": {
                    "function_name": "calculate_total",
                    "signature": {"parameters": [{"name": "items", "type": "list"}, {"name": "tax_rate", "type": "float"}]},
                    "variable_names": [{"name": "items", "preserve_exact_name": True}, {"name": "tax_rate", "preserve_exact_name": True}, 
                                      {"name": "subtotal", "preserve_exact_name": True}, {"name": "tax", "preserve_exact_name": True}, 
                                      {"name": "total", "preserve_exact_name": True}],
                    "control_flow": "if-else structure"
                },
                "example_code": """def calculate_total(items, tax_rate):
    subtotal = sum(items)
    tax = subtotal * tax_rate
    total = subtotal + tax
    return total""",
                "explanation": "CRITICAL: Uses EXACT variable names 'items', 'tax_rate', 'subtotal', 'tax', 'total' - NOT 'prices', 'rate', 'sum', 'fee', 'result', etc. Variable name matching is essential for high similarity scores."
            },
            {
                "specification": {
                    "function_name": "filter_even",
                    "signature": {"parameters": [{"name": "numbers", "type": "list"}]},
                    "variable_names": [{"name": "numbers", "preserve_exact_name": True}, {"name": "result", "preserve_exact_name": True}, 
                                      {"name": "num", "preserve_exact_name": True}],
                    "control_flow": "for loop with if condition"
                },
                "example_code": """def filter_even(numbers):
    result = []
    for num in numbers:
        if num % 2 == 0:
            result.append(num)
    return result""",
                "explanation": "CRITICAL: Uses EXACT names 'numbers' (NOT 'nums'), 'result' (NOT 'output' or 'res'), 'num' (NOT 'n', 'item', 'i', 'x'). Control flow matches: for loop with nested if statement."
            },
            {
                "specification": {
                    "function_name": "process_data",
                    "signature": {"parameters": [{"name": "data", "type": "dict"}, {"name": "key", "type": "str"}]},
                    "variable_names": ["data", "key", "value", "result"],
                    "control_flow": "if statement with return"
                },
                "example_code": """def process_data(data, key):
    if key in data:
        value = data[key]
        result = value * 2
        return result
    return None""",
                "explanation": "Notice: Preserves exact variable names 'data', 'key', 'value', 'result' - critical for similarity matching"
            },
            {
                "specification": {
                    "function_name": "find_max",
                    "signature": {"parameters": [{"name": "numbers", "type": "list"}]},
                    "variable_names": ["numbers", "max_val", "num"],
                    "control_flow": "for loop"
                },
                "example_code": """def find_max(numbers):
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val""",
                "explanation": "Notice: Uses 'numbers', 'max_val', 'num' exactly as specified - NOT 'nums', 'maximum', 'n'"
            }
        ]
    
    def _load_spec_examples(self) -> List[Dict[str, Any]]:
        """Load few-shot examples demonstrating good specification generation"""
        return [
            {
                "original_code": """def calculate_discount(price, discount_rate):
    if price < 0:
        raise ValueError("Price cannot be negative")
    if discount_rate < 0 or discount_rate > 1:
        raise ValueError("Discount rate must be between 0 and 1")
    discounted_price = price * (1 - discount_rate)
    return round(discounted_price, 2)""",
                "example_spec": {
                    "function_name": "calculate_discount",
                    "signature": {
                        "parameters": [
                            {"name": "price", "type": "float", "description": "Original price", "constraints": "Must be non-negative"},
                            {"name": "discount_rate", "type": "float", "description": "Discount rate as decimal", "constraints": "Must be between 0 and 1"}
                        ],
                        "return_type": "float"
                    },
                    "english_summary": "Calculates discounted price after applying discount rate. Validates inputs and rounds result to 2 decimal places.",
                    "user_stories": [
                        {
                            "id": "US1",
                            "priority": "P1",
                            "title": "Calculate valid discount",
                            "narrative": "User provides valid price and discount rate",
                            "acceptance": [
                                {"given": "price >= 0 and 0 <= discount_rate <= 1", "when": "function is called", "then": "returns rounded discounted price"}
                            ]
                        },
                        {
                            "id": "US2",
                            "priority": "P1",
                            "title": "Reject negative price",
                            "narrative": "User provides negative price",
                            "acceptance": [
                                {"given": "price < 0", "when": "function is called", "then": "raises ValueError with message about negative price"}
                            ]
                        }
                    ],
                    "error_handling": {
                        "exceptions": [
                            {"type": "ValueError", "condition": "price < 0", "message": "Price cannot be negative"},
                            {"type": "ValueError", "condition": "discount_rate < 0 or discount_rate > 1", "message": "Discount rate must be between 0 and 1"}
                        ]
                    },
                    "variable_names": ["price", "discount_rate", "discounted_price"]
                },
                "explanation": "Key: Captures exact variable names, error conditions with exact messages, return value rounding, and maps each behavior to user stories"
            },
            {
                "original_code": """def filter_by_length(words, min_length):
    result = []
    for word in words:
        if len(word) >= min_length:
            result.append(word)
    return result""",
                "example_spec": {
                    "function_name": "filter_by_length",
                    "signature": {
                        "parameters": [
                            {"name": "words", "type": "list", "description": "List of words to filter"},
                            {"name": "min_length", "type": "int", "description": "Minimum length threshold"}
                        ],
                        "return_type": "list"
                    },
                    "english_summary": "Filters words by minimum length, returning only words that meet or exceed the threshold.",
                    "user_stories": [
                        {
                            "id": "US1",
                            "priority": "P1",
                            "title": "Filter words by length",
                            "narrative": "User provides list of words and minimum length",
                            "acceptance": [
                                {"given": "list of words and min_length", "when": "function is called", "then": "returns list containing only words with length >= min_length"}
                            ]
                        }
                    ],
                    "variable_names": ["words", "min_length", "result", "word"],
                    "control_flow": "for loop with if condition"
                },
                "explanation": "Key: Preserves exact variable names (words, min_length, result, word), documents control flow pattern, and creates clear user story"
            }
        ]
    
    def _load_test_examples(self) -> List[Dict[str, Any]]:
        """Load few-shot examples demonstrating good test generation"""
        return [
            {
                "function_name": "calculate_discount",
                "specification": {
                    "signature": {"parameters": [{"name": "price", "type": "float"}, {"name": "discount_rate", "type": "float"}]},
                    "error_handling": {
                        "exceptions": [
                            {"type": "ValueError", "condition": "price < 0", "message": "Price cannot be negative"}
                        ]
                    }
                },
                "example_tests": [
                    {
                        "test_name": "test_valid_discount",
                        "inputs": {"price": 100.0, "discount_rate": 0.1},
                        "expected_output": 90.0,
                        "description": "Valid discount calculation"
                    },
                    {
                        "test_name": "test_negative_price_error",
                        "inputs": {"price": -10.0, "discount_rate": 0.1},
                        "expected_exception": "ValueError",
                        "description": "Rejects negative price"
                    }
                ],
                "explanation": "Key: Tests cover both success path and error path, use exact parameter names, and match expected outputs/exceptions"
            },
            {
                "function_name": "filter_by_length",
                "specification": {
                    "signature": {"parameters": [{"name": "words", "type": "list"}, {"name": "min_length", "type": "int"}]}
                },
                "example_tests": [
                    {
                        "test_name": "test_filter_short_words",
                        "inputs": {"words": ["a", "ab", "abc", "abcd"], "min_length": 3},
                        "expected_output": ["abc", "abcd"],
                        "description": "Filters words below minimum length"
                    },
                    {
                        "test_name": "test_empty_list",
                        "inputs": {"words": [], "min_length": 3},
                        "expected_output": [],
                        "description": "Handles empty input list"
                    }
                ],
                "explanation": "Key: Tests cover normal case and edge case (empty list), use exact parameter names from spec"
            },
            {
                "function_name": "process_numbers",
                "specification": {
                    "signature": {"parameters": [{"name": "numbers", "type": "list"}]},
                    "return_type": "dict"
                },
                "example_tests": [
                    {
                        "test_name": "test_process_valid_numbers",
                        "inputs": {"numbers": [1, 2, 3, 4, 5]},
                        "expected_output": {
                            "count": 5,
                            "sum": 15,
                            "average": 3.0,
                            "min": 1,
                            "max": 5,
                            "sorted": [1, 2, 3, 4, 5]
                        },
                        "description": "Processes list of numbers and returns statistics dictionary"
                    },
                    {
                        "test_name": "test_empty_list",
                        "inputs": {"numbers": []},
                        "expected_output": {
                            "count": 0,
                            "sum": 0,
                            "average": 0.0,
                            "min": None,
                            "max": None,
                            "sorted": []
                        },
                        "description": "Handles empty input list"
                    }
                ],
                "explanation": "Key: For functions returning dictionaries, include ALL expected keys with exact values. Use exact parameter names. Cover edge cases like empty lists."
            },
            {
                "function_name": "filter_data",
                "specification": {
                    "signature": {"parameters": [{"name": "data", "type": "list"}, {"name": "condition_func", "type": "callable"}]}
                },
                "example_tests": [
                    {
                        "test_name": "test_filter_even_numbers",
                        "inputs": {
                            "data": [1, 2, 3, 4, 5],
                            "condition_func": "lambda x: x % 2 == 0"
                        },
                        "expected_output": [2, 4],
                        "description": "Filters data using lambda function condition"
                    },
                    {
                        "test_name": "test_filter_empty_result",
                        "inputs": {
                            "data": [1, 3, 5],
                            "condition_func": "lambda x: x % 2 == 0"
                        },
                        "expected_output": [],
                        "description": "Returns empty list when no items match condition"
                    }
                ],
                "explanation": "Key: For callable parameters (like lambda functions), provide them as strings that can be evaluated. Use exact parameter names. Cover cases where filter returns empty results."
            }
        ]
    
    def add_examples_to_prompt(self, prompt: str, specification: Dict[str, Any]) -> str:
        """Add few-shot examples to code generation prompt"""
        examples = self._select_adaptive_examples(specification, self.code_examples, task_type="code")
        examples_section = "\n\nFEW-SHOT EXAMPLES (LEARN FROM THESE):\n"
        examples_section += "The following examples show CORRECT code regeneration that preserves variable names:\n\n"
        
        for i, example in enumerate(examples[:2], 1):
            examples_section += f"Example {i}:\n"
            examples_section += f"Specification: {example['specification']['function_name']} with variables: {', '.join(example['specification'].get('variable_names', []))}\n"
            examples_section += f"Generated Code:\n```python\n{example['example_code']}\n```\n"
            examples_section += f"Key: {example['explanation']}\n\n"
        
        examples_section += "YOUR TASK: Generate code following these examples - use EXACT variable names from specification.\n"
        
        # Insert examples before the critical requirements section
        if "CRITICAL REQUIREMENTS FOR" in prompt:
            # Handle "CRITICAL REQUIREMENTS FOR CLASS METHOD" case
            import re
            prompt = re.sub(r"(CRITICAL REQUIREMENTS FOR [^\n]+)", examples_section + "\n\\1", prompt, count=1)
        elif "CRITICAL REQUIREMENTS" in prompt:
            prompt = prompt.replace("CRITICAL REQUIREMENTS", examples_section + "\nCRITICAL REQUIREMENTS", 1)
        else:
            # If no CRITICAL REQUIREMENTS section, append examples before the end
            prompt += examples_section
        
        return prompt
    
    def add_spec_examples_to_prompt(self, prompt: str, source_code: str) -> str:
        """Add few-shot examples to specification generation prompt"""
        examples = self._select_adaptive_examples_for_spec(source_code, self.spec_examples)
        examples_section = "\n\nFEW-SHOT EXAMPLES (LEARN FROM THESE SPECIFICATIONS):\n"
        examples_section += "The following examples show CORRECT specification generation that captures all details:\n\n"
        
        for i, example in enumerate(examples[:2], 1):
            examples_section += f"Example {i}:\n"
            examples_section += f"Original Code:\n```python\n{example['original_code']}\n```\n"
            examples_section += f"Generated Specification (key fields):\n"
            spec = example['example_spec']
            examples_section += f"  - Function: {spec.get('function_name')}\n"
            examples_section += f"  - Variables: {', '.join(spec.get('variable_names', []))}\n"
            if spec.get('user_stories'):
                examples_section += f"  - User Stories: {len(spec['user_stories'])} stories with Given/When/Then\n"
            if spec.get('error_handling'):
                examples_section += f"  - Error Handling: Documented with exact exception types and messages\n"
            examples_section += f"Key: {example['explanation']}\n\n"
        
        examples_section += "YOUR TASK: Generate specification following these examples - capture ALL variable names, control flow, error handling, and map behaviors to user stories.\n"
        
        if "SPEC CHARTER" in prompt:
            prompt = prompt.replace("SPEC CHARTER", examples_section + "\nSPEC CHARTER")
        elif "Generate a detailed specification" in prompt:
            prompt = prompt.replace("Generate a detailed specification", examples_section + "\nGenerate a detailed specification")
        else:
            prompt = examples_section + "\n" + prompt
        
        return prompt
    
    def add_test_examples_to_prompt(self, prompt: str, specification: Dict[str, Any], function_name: str) -> str:
        """Add few-shot examples to test generation prompt"""
        examples = self._select_adaptive_examples_for_test(specification, function_name, self.test_examples)
        examples_section = "\n\nFEW-SHOT EXAMPLES (LEARN FROM THESE TEST CASES):\n"
        examples_section += "The following examples show CORRECT test generation that covers all behaviors:\n\n"
        
        for i, example in enumerate(examples[:2], 1):
            examples_section += f"Example {i}:\n"
            examples_section += f"Function: {example['function_name']}\n"
            examples_section += f"Test Cases:\n"
            for test in example['example_tests']:
                examples_section += f"  - {test['test_name']}: inputs={test.get('inputs', {})}, "
                if 'expected_output' in test:
                    examples_section += f"expected_output={test['expected_output']}"
                elif 'expected_exception' in test:
                    examples_section += f"expected_exception={test['expected_exception']}"
                examples_section += "\n"
            examples_section += f"Key: {example['explanation']}\n\n"
        
        examples_section += "YOUR TASK: Generate tests following these examples - use EXACT parameter names from specification, cover all paths including errors.\n"
        
        if "### Output format" in prompt:
            prompt = prompt.replace("### Output format", examples_section + "\n### Output format")
        else:
            prompt = examples_section + "\n" + prompt
        
        return prompt
    
    def _select_adaptive_examples(self, specification: Dict[str, Any], examples: List[Dict[str, Any]], task_type: str = "code") -> List[Dict[str, Any]]:
        """Adaptively select examples based on code similarity/patterns"""
        if not examples:
            return []
        
        # Handle variable_names which can be a list of strings or list of dicts
        var_names_raw = specification.get('variable_names', [])
        if var_names_raw and isinstance(var_names_raw[0], dict):
            spec_vars = set(v.get('name', '') for v in var_names_raw if isinstance(v, dict) and v.get('name'))
        else:
            spec_vars = set(str(v) for v in var_names_raw)
        
        spec_control_flow = specification.get('control_flow', '')
        
        scored_examples = []
        for example in examples:
            score = 0.0
            example_spec = example.get('specification', {})
            
            # Handle variable_names which can be a list of strings or list of dicts
            example_var_names_raw = example_spec.get('variable_names', [])
            if example_var_names_raw and isinstance(example_var_names_raw[0], dict):
                example_vars = set(v.get('name', '') for v in example_var_names_raw if isinstance(v, dict) and v.get('name'))
            else:
                example_vars = set(str(v) for v in example_var_names_raw)
            
            example_control_flow = example_spec.get('control_flow', '')
            
            if spec_vars and example_vars:
                var_overlap = len(spec_vars & example_vars) / max(len(spec_vars | example_vars), 1)
                score += var_overlap * 0.4
            
            if spec_control_flow and example_control_flow:
                if spec_control_flow.lower() in example_control_flow.lower() or example_control_flow.lower() in spec_control_flow.lower():
                    score += 0.3
            
            if specification.get('function_name') and example_spec.get('function_name'):
                if any(keyword in specification['function_name'].lower() for keyword in ['filter', 'find', 'calculate', 'process']):
                    if any(keyword in example_spec['function_name'].lower() for keyword in ['filter', 'find', 'calculate', 'process']):
                        score += 0.3
            
            scored_examples.append((score, example))
        
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored_examples]
    
    def _select_adaptive_examples_for_spec(self, source_code: str, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Adaptively select spec examples based on code characteristics"""
        if not examples:
            return examples
        
        try:
            tree = ast.parse(source_code)
            has_loops = any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
            has_conditionals = any(isinstance(node, ast.If) for node in ast.walk(tree))
            has_exceptions = any(isinstance(node, ast.Raise) for node in ast.walk(tree))
            
            scored_examples = []
            for example in examples:
                score = 0.0
                example_code = example.get('original_code', '')
                
                try:
                    example_tree = ast.parse(example_code)
                    example_has_loops = any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(example_tree))
                    example_has_conditionals = any(isinstance(node, ast.If) for node in ast.walk(example_tree))
                    example_has_exceptions = any(isinstance(node, ast.Raise) for node in ast.walk(example_tree))
                    
                    if has_loops == example_has_loops:
                        score += 0.3
                    if has_conditionals == example_has_conditionals:
                        score += 0.3
                    if has_exceptions == example_has_exceptions:
                        score += 0.4
                except:
                    pass
                
                scored_examples.append((score, example))
            
            scored_examples.sort(key=lambda x: x[0], reverse=True)
            return [ex for _, ex in scored_examples]
        except:
            return examples
    
    def _select_adaptive_examples_for_test(self, specification: Dict[str, Any], function_name: str, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Adaptively select test examples based on spec characteristics"""
        if not examples:
            return examples
        
        # Handle error_handling which can be a dict or list
        error_handling = specification.get('error_handling', {})
        if isinstance(error_handling, dict):
            has_error_handling = bool(error_handling.get('exceptions'))
        elif isinstance(error_handling, list):
            has_error_handling = len(error_handling) > 0
        else:
            has_error_handling = False
        
        num_params = len(specification.get('signature', {}).get('parameters', []))
        
        scored_examples = []
        for example in examples:
            score = 0.0
            example_spec = example.get('specification', {})
            
            # Handle error_handling which can be a dict or list
            example_error_handling = example_spec.get('error_handling', {})
            if isinstance(example_error_handling, dict):
                example_has_errors = bool(example_error_handling.get('exceptions'))
            elif isinstance(example_error_handling, list):
                example_has_errors = len(example_error_handling) > 0
            else:
                example_has_errors = False
            
            example_num_params = len(example_spec.get('signature', {}).get('parameters', []))
            
            if has_error_handling == example_has_errors:
                score += 0.5
            
            if abs(num_params - example_num_params) <= 1:
                score += 0.3
            
            if function_name and example.get('function_name'):
                if any(keyword in function_name.lower() for keyword in ['filter', 'find', 'calculate', 'process']):
                    if any(keyword in example['function_name'].lower() for keyword in ['filter', 'find', 'calculate', 'process']):
                        score += 0.2
            
            scored_examples.append((score, example))
        
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored_examples]
    
    def add_variable_name_enforcement(self, prompt: str, specification: Dict[str, Any]) -> str:
        """Add strong enforcement for variable name preservation"""
        var_names = specification.get('variable_names', [])
        if not var_names:
            return prompt
        
        enforcement = "\n\nVARIABLE NAME ENFORCEMENT:\n"
        enforcement += "The following variables MUST appear in your code with EXACTLY these names:\n"
        for var in var_names[:10]:  # Limit to first 10
            if isinstance(var, dict):
                name = var.get('name', '')
            else:
                name = str(var)
            if name:
                enforcement += f"  - {name}\n"
        
        enforcement += "\nDO NOT use synonyms or alternative names. For example:\n"
        enforcement += "  - If spec says 'result', use 'result' NOT 'output', 'res', 'value'\n"
        enforcement += "  - If spec says 'data', use 'data' NOT 'items', 'list', 'arr'\n"
        enforcement += "  - If spec says 'i', use 'i' NOT 'idx', 'index', 'j'\n"
        enforcement += "Variable name matching is CRITICAL for similarity scores.\n"
        
        if "CRITICAL REQUIREMENTS" in prompt:
            prompt = prompt.replace("CRITICAL REQUIREMENTS", enforcement + "\nCRITICAL REQUIREMENTS", 1)
        elif "CRITICAL REQUIREMENTS FOR" in prompt:
            # Handle "CRITICAL REQUIREMENTS FOR CLASS METHOD" case
            import re
            prompt = re.sub(r"(CRITICAL REQUIREMENTS FOR [^\n]+)", enforcement + "\n\\1", prompt, count=1)
        else:
            # If no CRITICAL REQUIREMENTS section, append enforcement before the end
            prompt += enforcement
        
        return prompt

