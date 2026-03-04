"""
Enhanced Test Generation with Branch Coverage Targeting
Improves test generation to specifically target missing branches and lines
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import ast
import json


class EnhancedTestGenerator:
    """Enhanced test generation targeting missing coverage"""
    
    def generate_targeted_tests(
        self,
        specification: Dict[str, Any],
        missing_branches: List[Dict[str, Any]],
        missing_lines: List[int],
        original_code: str,
        function_name: str
    ) -> List[Dict[str, Any]]:
        """
        Generate tests specifically targeting missing branches and lines.
        
        Strategy:
        1. Analyze missing branches to understand conditions
        2. Generate test inputs that trigger those branches
        3. Target missing lines with specific test cases
        """
        targeted_tests = []
        
        # Analyze missing branches
        branch_tests = self._generate_branch_targeting_tests(
            missing_branches, original_code, function_name
        )
        targeted_tests.extend(branch_tests)
        
        # Analyze missing lines
        line_tests = self._generate_line_targeting_tests(
            missing_lines, original_code, function_name
        )
        targeted_tests.extend(line_tests)
        
        return targeted_tests
    
    def _generate_branch_targeting_tests(
        self,
        missing_branches: List[Dict[str, Any]],
        original_code: str,
        function_name: str
    ) -> List[Dict[str, Any]]:
        """Generate tests targeting specific missing branches"""
        tests = []
        
        try:
            tree = ast.parse(original_code)
            
            # Find conditionals that correspond to missing branches
            for branch_info in missing_branches:
                line_no = branch_info.get('line', 0)
                if line_no == 0:
                    continue
                
                # Find the conditional at this line
                conditional = self._find_conditional_at_line(tree, line_no)
                if conditional:
                    # Generate test that triggers this branch
                    test = self._create_branch_test(conditional, original_code, function_name)
                    if test:
                        tests.append(test)
        
        except Exception:
            pass
        
        return tests
    
    def _find_conditional_at_line(self, tree: ast.AST, line_no: int) -> Optional[ast.If]:
        """Find the if statement at a specific line"""
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                if node.lineno == line_no or (hasattr(node, 'end_lineno') and node.lineno <= line_no <= node.end_lineno):
                    return node
        return None
    
    def _create_branch_test(
        self,
        conditional: ast.If,
        original_code: str,
        function_name: str
    ) -> Optional[Dict[str, Any]]:
        """Create a test case that triggers a specific branch"""
        # Extract condition
        condition_str = ast.unparse(conditional.test) if hasattr(ast, 'unparse') else str(conditional.test)
        
        # Generate input that satisfies condition
        # This is simplified - full implementation would use constraint solving
        test_inputs = self._infer_test_inputs_from_condition(condition_str, original_code)
        
        if test_inputs:
            return {
                'test_name': f'test_branch_{conditional.lineno}',
                'inputs': test_inputs,
                'description': f'Targets branch at line {conditional.lineno} with condition: {condition_str}',
                'target_branch': conditional.lineno
            }
        
        return None
    
    def _infer_test_inputs_from_condition(self, condition: str, code: str) -> Dict[str, Any]:
        """Infer test inputs from a condition (simplified heuristic)"""
        inputs = {}
        
        # Simple heuristics for common conditions
        if '>' in condition:
            # Extract variable and value
            parts = condition.split('>')
            if len(parts) == 2:
                var = parts[0].strip()
                try:
                    val = int(parts[1].strip())
                    inputs[var] = val + 1  # Value that satisfies >
                except:
                    pass
        
        elif '<' in condition:
            parts = condition.split('<')
            if len(parts) == 2:
                var = parts[0].strip()
                try:
                    val = int(parts[1].strip())
                    inputs[var] = val - 1  # Value that satisfies <
                except:
                    pass
        
        elif '==' in condition:
            parts = condition.split('==')
            if len(parts) == 2:
                var = parts[0].strip()
                val = parts[1].strip().strip('"\'')
                try:
                    inputs[var] = int(val)
                except:
                    inputs[var] = val
        
        elif 'is None' in condition or '== None' in condition:
            var = condition.split()[0]
            inputs[var] = None
        
        elif 'is not None' in condition or '!= None' in condition:
            var = condition.split()[0]
            inputs[var] = []  # Non-None value
        
        return inputs
    
    def _generate_line_targeting_tests(
        self,
        missing_lines: List[int],
        original_code: str,
        function_name: str
    ) -> List[Dict[str, Any]]:
        """Generate tests targeting specific missing lines"""
        tests = []
        
        try:
            tree = ast.parse(original_code)
            lines = original_code.splitlines()
            
            for line_no in missing_lines[:5]:  # Limit to 5
                if line_no > len(lines):
                    continue
                
                line_content = lines[line_no - 1].strip()
                
                # Skip empty lines, comments, etc.
                if not line_content or line_content.startswith('#'):
                    continue
                
                # Find what this line does
                node = self._find_node_at_line(tree, line_no)
                if node:
                    test = self._create_line_test(node, line_no, original_code, function_name)
                    if test:
                        tests.append(test)
        
        except Exception:
            pass
        
        return tests
    
    def _find_node_at_line(self, tree: ast.AST, line_no: int) -> Optional[ast.AST]:
        """Find AST node at a specific line"""
        for node in ast.walk(tree):
            if hasattr(node, 'lineno') and node.lineno == line_no:
                return node
        return None
    
    def _create_line_test(
        self,
        node: ast.AST,
        line_no: int,
        original_code: str,
        function_name: str
    ) -> Optional[Dict[str, Any]]:
        """Create test case targeting a specific line"""
        # Simplified: generate test that would execute this line
        # Full implementation would analyze what inputs are needed
        
        return {
            'test_name': f'test_line_{line_no}',
            'inputs': {},  # Would be inferred
            'description': f'Targets execution of line {line_no}',
            'target_line': line_no
        }
    
    def enhance_test_prompt_with_coverage_targets(
        self,
        base_prompt: str,
        missing_branches: List[Dict[str, Any]],
        missing_lines: List[int],
        current_coverage: float
    ) -> str:
        """Enhance test generation prompt with coverage targets"""
        if current_coverage >= 0.8:
            return base_prompt  # Good coverage, no need to enhance
        
        coverage_section = "\n\nCOVERAGE TARGETS (CRITICAL - Current coverage is low):\n"
        coverage_section += f"Current branch coverage: {current_coverage:.1%}\n"
        coverage_section += "You MUST generate tests that target the following:\n\n"
        
        if missing_branches:
            coverage_section += "MISSING BRANCHES (must be covered):\n"
            for branch in missing_branches[:10]:
                line = branch.get('line', 0)
                detail = branch.get('detail', '')
                coverage_section += f"  - Line {line}: {detail}\n"
            coverage_section += "\nGenerate test cases that specifically trigger these branches.\n\n"
        
        if missing_lines:
            coverage_section += "MISSING LINES (must be executed):\n"
            coverage_section += f"  - Lines: {', '.join(map(str, missing_lines[:15]))}\n"
            coverage_section += "\nGenerate test cases that execute these specific lines.\n"
        
        coverage_section += "\nPRIORITY: Coverage is more important than number of tests. "
        coverage_section += "Generate fewer, more targeted tests that hit these specific branches/lines.\n"
        
        return base_prompt + coverage_section

