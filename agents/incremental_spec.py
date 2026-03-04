"""
Incremental Specification Building
Builds specifications incrementally by analyzing and adding one aspect at a time
"""

from typing import Dict, List, Any, Optional, Set
import ast
import json


class IncrementalSpecBuilder:
    """Builds specifications incrementally for better accuracy"""
    
    def __init__(self):
        self.spec_layers = [
            'signature',
            'control_flow',
            'data_flow',
            'edge_cases',
            'examples',
            'error_handling'
        ]
    
    def build_incrementally(
        self,
        code: str,
        function_name: str,
        existing_spec: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build specification layer by layer"""
        spec = existing_spec.copy() if existing_spec else {}
        
        try:
            if not code or not isinstance(code, str) or len(code.strip()) < 10:
                return spec
            
            tree = ast.parse(code)
            func_node = self._find_function(tree, function_name)
            if not func_node:
                return spec
            
            # Layer 1: Signature (always first)
            if 'signature' not in spec or not spec.get('signature'):
                spec['signature'] = self._extract_signature(func_node)
            
            # Layer 2: Control flow
            if 'control_flow' not in spec or not spec.get('control_flow'):
                spec['control_flow'] = self._extract_control_flow(func_node)
            
            # Layer 3: Data flow
            if 'variable_names' not in spec or not spec.get('variable_names'):
                spec['variable_names'] = self._extract_data_flow(func_node)
            
            # Layer 4: Return analysis
            if 'return_type' not in spec or not spec.get('return_type'):
                spec['return_type'] = self._extract_return_info(func_node)
            
            # Layer 5: Edge cases
            if 'edge_cases' not in spec or not spec.get('edge_cases'):
                spec['edge_cases'] = self._identify_edge_cases(func_node, code)
            
            # Layer 6: Error handling
            if 'error_handling' not in spec or not spec.get('error_handling'):
                spec['error_handling'] = self._extract_error_handling(func_node)
            
            return spec
        
        except Exception as e:
            print(f"        Error in incremental spec building: {e}")
            return spec
    
    def _find_function(self, tree: ast.AST, function_name: str) -> Optional[ast.FunctionDef]:
        """Find function node by name"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return node
        return None
    
    def _extract_signature(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract function signature"""
        params = []
        for arg in func_node.args.args:
            param_info = {'name': arg.arg}
            if arg.annotation:
                param_info['type'] = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation)
            params.append(param_info)
        
        return_type = None
        if func_node.returns:
            return_type = ast.unparse(func_node.returns) if hasattr(ast, 'unparse') else str(func_node.returns)
        
        return {
            'function_name': func_node.name,
            'parameters': params,
            'return_type': return_type
        }
    
    def _extract_control_flow(self, func_node: ast.FunctionDef) -> str:
        """Extract control flow structure"""
        flow_elements = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.If):
                flow_elements.append("if-else branches")
            elif isinstance(node, ast.For):
                flow_elements.append("for loop")
            elif isinstance(node, ast.While):
                flow_elements.append("while loop")
            elif isinstance(node, ast.Try):
                flow_elements.append("try-except block")
            elif isinstance(node, ast.Return):
                if node.value:
                    flow_elements.append("returns value")
                else:
                    flow_elements.append("returns None")
        
        return "; ".join(sorted(set(flow_elements)))
    
    def _extract_data_flow(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract variable names and data flow"""
        variables = []
        seen = set()
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        if var_name not in seen and not var_name.startswith('_'):
                            seen.add(var_name)
                            variables.append({
                                'name': var_name,
                                'exact_match_required': True
                            })
        
        return variables
    
    def _extract_return_info(self, func_node: ast.FunctionDef) -> str:
        """Extract return type information"""
        returns = [n for n in ast.walk(func_node) if isinstance(n, ast.Return)]
        
        if not returns:
            return "None"
        
        return_types = set()
        for ret in returns:
            if ret.value:
                if isinstance(ret.value, ast.Dict):
                    return_types.add("dict")
                elif isinstance(ret.value, ast.List):
                    return_types.add("list")
                elif isinstance(ret.value, ast.Str):
                    return_types.add("str")
                elif isinstance(ret.value, ast.Num):
                    return_types.add("int/float")
                else:
                    return_types.add("object")
            else:
                return_types.add("None")
        
        if len(return_types) == 1:
            return list(return_types)[0]
        else:
            return f"Union[{', '.join(sorted(return_types))}]"
    
    def _identify_edge_cases(self, func_node: ast.FunctionDef, code: str) -> List[str]:
        """Identify potential edge cases"""
        edge_cases = []
        
        # Check for None comparisons
        has_none_check = any(
            isinstance(node, ast.Compare) and
            any(isinstance(cmp, ast.Constant) and cmp.value is None for cmp in node.comparators)
            for node in ast.walk(func_node)
        )
        if has_none_check:
            edge_cases.append("handles None input")
        
        # Check for empty collections
        has_empty_check = any(
            (isinstance(node, ast.If) and
             isinstance(node.test, ast.Compare) and
             any(isinstance(op, ast.Eq) for op in node.test.ops))
            for node in ast.walk(func_node)
        )
        if has_empty_check:
            edge_cases.append("handles empty collections")
        
        # Check for division operations
        has_division = any(
            isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv))
            for node in ast.walk(func_node)
        )
        if has_division:
            edge_cases.append("handles division by zero")
        
        return edge_cases
    
    def _extract_error_handling(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract error handling information"""
        exceptions = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    if handler.type:
                        exc_type = ast.unparse(handler.type) if hasattr(ast, 'unparse') else str(handler.type)
                        exceptions.append({
                            'type': exc_type,
                            'handled': True
                        })
        
        return {
            'exceptions': exceptions,
            'has_error_handling': len(exceptions) > 0
        }
    
    def enhance_existing_spec(
        self,
        existing_spec: Dict[str, Any],
        code: str,
        function_name: str
    ) -> Dict[str, Any]:
        """Enhance an existing specification with missing layers"""
        return self.build_incrementally(code, function_name, existing_spec)

