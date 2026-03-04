"""
Enhanced Specification Generator for Complex Functions
Generates detailed specifications with comprehensive English descriptions,
step-by-step logic flow, and detailed variable usage tracking
"""

import ast
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict


class EnhancedSpecGenerator:
    """Generates enhanced specifications with detailed documentation for complex functions"""
    
    def __init__(self):
        pass
    
    def enhance_specification(self, specification: Dict[str, Any], code: str, function_name: str) -> Dict[str, Any]:
        """Enhance a specification with detailed documentation"""
        try:
            tree = ast.parse(code)
            func_node = self._find_function(tree, function_name)
            if not func_node:
                return specification
            
            # Enhance with detailed English description
            specification['detailed_english_description'] = self._generate_detailed_description(
                func_node, specification, code
            )
            
            # Add step-by-step logic flow
            specification['detailed_step_by_step_logic'] = self._extract_step_by_step_logic(
                func_node, code
            )
            
            # Enhance variable usage documentation
            specification['detailed_variable_usage'] = self._extract_detailed_variable_usage(
                func_node, code
            )
            
            # Enhance control flow with nesting
            specification['detailed_control_flow'] = self._extract_detailed_control_flow(
                func_node
            )
            
            return specification
        
        except Exception as e:
            return specification
    
    def _find_function(self, tree: ast.AST, function_name: str) -> Optional[ast.FunctionDef]:
        """Find function node by name"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return node
        return None
    
    def _generate_detailed_description(
        self, func_node: ast.FunctionDef, specification: Dict[str, Any], code: str
    ) -> str:
        """Generate comprehensive English description (5-10 sentences for complex functions)"""
        parts = []
        
        # Function purpose
        func_name = func_node.name
        params = [arg.arg for arg in func_node.args.args if arg.arg != 'self']
        
        if params:
            parts.append(
                f"The function {func_name} takes {len(params)} parameter(s): {', '.join(params)}."
            )
        else:
            parts.append(f"The function {func_name} takes no parameters.")
        
        # Return behavior
        returns = [n for n in ast.walk(func_node) if isinstance(n, ast.Return)]
        if returns:
            parts.append(
                f"The function returns a value using {len(returns)} return statement(s)."
            )
        else:
            parts.append("The function does not explicitly return a value.")
        
        # Control flow summary
        if_stmts = [n for n in ast.walk(func_node) if isinstance(n, ast.If)]
        loops = [n for n in ast.walk(func_node) if isinstance(n, (ast.For, ast.While))]
        try_blocks = [n for n in ast.walk(func_node) if isinstance(n, ast.Try)]
        
        flow_parts = []
        if if_stmts:
            flow_parts.append(f"{len(if_stmts)} conditional branch(es)")
        if loops:
            flow_parts.append(f"{len(loops)} loop(s)")
        if try_blocks:
            flow_parts.append(f"{len(try_blocks)} error handling block(s)")
        
        if flow_parts:
            parts.append(
                f"The function contains " + ", ".join(flow_parts) + "."
            )
        
        # Variable usage
        variables = set()
        for node in ast.walk(func_node):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                variables.add(node.id)
        
        if variables:
            var_list = list(variables)[:5]
            parts.append(
                f"The function manipulates variables: {', '.join(var_list)}"
                + (f" and {len(variables) - 5} more" if len(variables) > 5 else "") + "."
            )
        
        # Side effects
        side_effects = specification.get('side_effects', [])
        if side_effects:
            if isinstance(side_effects, list):
                parts.append(f"Side effects include: {', '.join(side_effects[:3])}.")
            else:
                parts.append(f"Side effects: {side_effects}.")
        
        # Error handling
        raises = [n for n in ast.walk(func_node) if isinstance(n, ast.Raise)]
        if raises:
            parts.append(f"The function may raise exceptions at {len(raises)} point(s).")
        
        # Algorithm summary
        if len(if_stmts) > 3 or len(loops) > 2:
            parts.append(
                "This is a complex function with multiple control flow paths and nested logic."
            )
        
        return " ".join(parts)
    
    def _extract_step_by_step_logic(
        self, func_node: ast.FunctionDef, code: str
    ) -> List[Dict[str, Any]]:
        """Extract detailed step-by-step logic flow"""
        steps = []
        step_num = 1
        
        def process_node(node, depth=0, step_counter=[1]):
            if isinstance(node, list):
                for item in node:
                    process_node(item, depth, step_counter)
                return
            
            indent = "  " * depth
            
            if isinstance(node, ast.Assign):
                targets = [self._ast_to_str(t) for t in node.targets]
                value = self._ast_to_str(node.value)
                steps.append({
                    'step': step_counter[0],
                    'type': 'assignment',
                    'description': f"Assign {value} to {', '.join(targets)}",
                    'indent_level': depth,
                    'line': getattr(node, 'lineno', None)
                })
                step_counter[0] += 1
            
            elif isinstance(node, ast.If):
                condition = self._ast_to_str(node.test)
                steps.append({
                    'step': step_counter[0],
                    'type': 'conditional',
                    'description': f"IF condition: {condition}",
                    'indent_level': depth,
                    'line': getattr(node, 'lineno', None)
                })
                step_counter[0] += 1
                process_node(node.body, depth + 1, step_counter)
                if node.orelse:
                    steps.append({
                        'step': step_counter[0],
                        'type': 'conditional_else',
                        'description': "ELSE branch",
                        'indent_level': depth,
                        'line': getattr(node, 'lineno', None)
                    })
                    process_node(node.orelse, depth + 1, step_counter)
            
            elif isinstance(node, ast.For):
                target = self._ast_to_str(node.target)
                iter_obj = self._ast_to_str(node.iter)
                steps.append({
                    'step': step_counter[0],
                    'type': 'loop',
                    'description': f"FOR loop: iterate over {iter_obj} as {target}",
                    'indent_level': depth,
                    'line': getattr(node, 'lineno', None)
                })
                step_counter[0] += 1
                process_node(node.body, depth + 1, step_counter)
            
            elif isinstance(node, ast.While):
                condition = self._ast_to_str(node.test)
                steps.append({
                    'step': step_counter[0],
                    'type': 'loop',
                    'description': f"WHILE loop: while {condition}",
                    'indent_level': depth,
                    'line': getattr(node, 'lineno', None)
                })
                step_counter[0] += 1
                process_node(node.body, depth + 1, step_counter)
            
            elif isinstance(node, ast.Return):
                value = self._ast_to_str(node.value) if node.value else "None"
                steps.append({
                    'step': step_counter[0],
                    'type': 'return',
                    'description': f"RETURN {value}",
                    'indent_level': depth,
                    'line': getattr(node, 'lineno', None)
                })
                step_counter[0] += 1
            
            elif isinstance(node, ast.Call):
                func_name = self._ast_to_str(node.func)
                steps.append({
                    'step': step_counter[0],
                    'type': 'call',
                    'description': f"CALL function: {func_name}",
                    'indent_level': depth,
                    'line': getattr(node, 'lineno', None)
                })
                step_counter[0] += 1
            
            elif isinstance(node, ast.Try):
                steps.append({
                    'step': step_counter[0],
                    'type': 'error_handling',
                    'description': "TRY block: begin error handling",
                    'indent_level': depth,
                    'line': getattr(node, 'lineno', None)
                })
                step_counter[0] += 1
                process_node(node.body, depth + 1, step_counter)
                for handler in node.handlers:
                    exc = handler.type
                    exc_name = self._ast_to_str(exc) if exc else "any exception"
                    steps.append({
                        'step': step_counter[0],
                        'type': 'error_handling',
                        'description': f"EXCEPT {exc_name}",
                        'indent_level': depth,
                        'line': getattr(handler, 'lineno', None)
                    })
                    step_counter[0] += 1
                    process_node(handler.body, depth + 1, step_counter)
            
            elif hasattr(node, 'body'):
                process_node(node.body, depth, step_counter)
        
        process_node(func_node.body, 0, [1])
        
        return steps
    
    def _extract_detailed_variable_usage(
        self, func_node: ast.FunctionDef, code: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extract detailed variable usage (where each variable is used)"""
        var_usage = defaultdict(list)
        lines = code.split('\n')
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Name):
                var_name = node.id
                line_num = getattr(node, 'lineno', None)
                ctx_type = 'assignment' if isinstance(node.ctx, ast.Store) else 'reference'
                
                context = "unknown"
                if line_num and line_num <= len(lines):
                    context_line = lines[line_num - 1].strip()
                    context = context_line[:100] if len(context_line) > 100 else context_line
                
                var_usage[var_name].append({
                    'line': line_num,
                    'context': context,
                    'usage_type': ctx_type,
                    'node_type': type(node).__name__
                })
        
        return dict(var_usage)
    
    def _extract_detailed_control_flow(
        self, func_node: ast.FunctionDef
    ) -> List[Dict[str, Any]]:
        """Extract detailed control flow with nesting levels"""
        flow_structure = []
        
        def extract_flow(node, depth=0, parent_type=None):
            if isinstance(node, list):
                for item in node:
                    extract_flow(item, depth, parent_type)
                return
            
            if isinstance(node, ast.If):
                condition = self._ast_to_str(node.test)
                flow_structure.append({
                    'type': 'if',
                    'condition': condition,
                    'nesting_level': depth,
                    'has_else': bool(node.orelse),
                    'line': getattr(node, 'lineno', None),
                    'children': []
                })
                parent = flow_structure[-1] if flow_structure else None
                extract_flow(node.body, depth + 1, 'if')
                if node.orelse:
                    extract_flow(node.orelse, depth + 1, 'else')
            
            elif isinstance(node, ast.For):
                target = self._ast_to_str(node.target)
                iter_obj = self._ast_to_str(node.iter)
                flow_structure.append({
                    'type': 'for',
                    'target': target,
                    'iterable': iter_obj,
                    'nesting_level': depth,
                    'line': getattr(node, 'lineno', None),
                    'children': []
                })
                extract_flow(node.body, depth + 1, 'for')
            
            elif isinstance(node, ast.While):
                condition = self._ast_to_str(node.test)
                flow_structure.append({
                    'type': 'while',
                    'condition': condition,
                    'nesting_level': depth,
                    'line': getattr(node, 'lineno', None),
                    'children': []
                })
                extract_flow(node.body, depth + 1, 'while')
            
            elif isinstance(node, ast.Try):
                flow_structure.append({
                    'type': 'try',
                    'nesting_level': depth,
                    'num_handlers': len(node.handlers),
                    'line': getattr(node, 'lineno', None),
                    'children': []
                })
                extract_flow(node.body, depth + 1, 'try')
                for handler in node.handlers:
                    exc = handler.type
                    exc_name = self._ast_to_str(exc) if exc else "any"
                    flow_structure.append({
                        'type': 'except',
                        'exception': exc_name,
                        'nesting_level': depth + 1,
                        'line': getattr(handler, 'lineno', None),
                        'children': []
                    })
                    extract_flow(handler.body, depth + 2, 'except')
            
            elif hasattr(node, 'body'):
                extract_flow(node.body, depth, parent_type)
        
        extract_flow(func_node.body, 0)
        return flow_structure
    
    def _ast_to_str(self, node: ast.AST) -> str:
        """Convert AST node to string representation"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Attribute):
            return f"{self._ast_to_str(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            func = self._ast_to_str(node.func)
            args = [self._ast_to_str(arg) for arg in node.args[:3]]
            return f"{func}({', '.join(args)})"
        elif isinstance(node, ast.Compare):
            left = self._ast_to_str(node.left)
            ops = [op.__class__.__name__ for op in node.ops]
            comparators = [self._ast_to_str(c) for c in node.comparators[:2]]
            return f"{left} {', '.join(ops)} {', '.join(comparators)}"
        elif isinstance(node, ast.BinOp):
            left = self._ast_to_str(node.left)
            op = type(node.op).__name__
            right = self._ast_to_str(node.right)
            return f"{left} {op} {right}"
        else:
            return type(node).__name__

