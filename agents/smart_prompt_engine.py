"""
Smart Prompt Engineering Agent
Advanced prompt engineering with context awareness and adaptive strategies
"""

import json
import re
import ast
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np


class SmartPromptEngine:
    """Advanced prompt engineering with adaptive strategies"""
    
    def __init__(self):
        self.prompt_templates = self._initialize_advanced_templates()
        self.context_analyzer = ContextAnalyzer()
        self.strategy_selector = StrategySelector()
        self.prompt_optimizer = PromptOptimizer()
    
    def generate_adaptive_prompt(self, 
                               code: str, 
                               context: Dict[str, Any], 
                               similarity_gaps: List[str] = None,
                               iteration: int = 1) -> str:
        """Generate adaptive prompt based on code characteristics and context"""
        
        # Analyze code characteristics
        code_analysis = self.context_analyzer.analyze_code_advanced(code)
        
        # Select appropriate strategy
        strategy = self.strategy_selector.select_strategy(code_analysis, context, iteration)
        
        # Generate base prompt
        base_prompt = self._generate_strategy_prompt(strategy, code, context)
        
        # Apply optimizations
        optimized_prompt = self.prompt_optimizer.optimize_prompt(
            base_prompt, code_analysis, similarity_gaps, iteration
        )
        
        return optimized_prompt
    
    def generate_iterative_prompt(self, 
                                 original_prompt: str, 
                                 feedback: Dict[str, Any], 
                                 iteration: int) -> str:
        """Generate improved prompt based on feedback from previous iterations"""
        
        # Analyze feedback patterns
        feedback_analysis = self._analyze_feedback_patterns(feedback, iteration)
        
        # Generate improvement strategies
        improvements = self._generate_improvement_strategies(feedback_analysis, iteration)
        
        # Apply improvements to prompt
        improved_prompt = self._apply_improvements(original_prompt, improvements, iteration)
        
        return improved_prompt
    
    def _initialize_advanced_templates(self) -> Dict[str, str]:
        """Initialize advanced prompt templates"""
        return {
            'simple_function': """
Analyze the following Python function and generate a comprehensive specification that would allow an LLM to regenerate the exact same code with 95%+ similarity.

Function to analyze:
```python
{code}
```

Generate a detailed specification including:
1. Function signature and parameters (exact names and types)
2. Return type and value
3. Internal logic and control flow
4. Variable names and their purposes
5. Comments and docstrings (exact text)
6. Error handling patterns
7. Edge cases and special conditions
8. Dependencies and imports
9. Side effects
10. Any other details needed for exact recreation

Format the specification as structured JSON with clear, detailed descriptions.
{context}
""",
            
            'complex_function': """
Analyze the following complex Python function and generate a comprehensive specification that would allow an LLM to regenerate the exact same code with 95%+ similarity.

Complex function to analyze:
```python
{code}
```

Generate a detailed specification including:
1. Function signature and parameters (exact names, types, defaults)
2. Return type and value
3. Internal logic and control flow (step-by-step)
4. Variable names and their purposes (all variables)
5. Comments and docstrings (exact text including formatting)
6. Error handling patterns and exceptions
7. Edge cases and special conditions
8. Dependencies and imports (all imports)
9. Side effects and state changes
10. Control flow structures (if/else, loops, try/catch)
11. Data structures and their usage
12. Algorithm steps and logic flow
13. Any other details needed for exact recreation

Pay special attention to:
- Exact variable names and their usage patterns
- Control flow structure and nesting
- Error handling and edge cases
- Comments and documentation
- Import statements and dependencies

Format the specification as structured JSON with clear, detailed descriptions.
{context}
""",
            
            'recursive_function': """
Analyze the following recursive Python function and generate a comprehensive specification that would allow an LLM to regenerate the exact same code with 95%+ similarity.

Recursive function to analyze:
```python
{code}
```

Generate a detailed specification including:
1. Function signature and parameters (exact names, types, defaults)
2. Return type and value
3. Base case conditions (exact logic)
4. Recursive case conditions (exact logic)
5. Internal logic and control flow
6. Variable names and their purposes
7. Comments and docstrings (exact text)
8. Error handling patterns
9. Edge cases and special conditions
10. Dependencies and imports
11. Side effects and state changes
12. Recursion depth and termination conditions
13. Any other details needed for exact recreation

Pay special attention to:
- Base case and recursive case logic
- Termination conditions
- Variable scoping and recursion state
- Error handling in recursive calls

Format the specification as structured JSON with clear, detailed descriptions.
{context}
""",
            
            'class_method': """
Analyze the following Python class method and generate a comprehensive specification that would allow an LLM to regenerate the exact same code with 95%+ similarity.

Class method to analyze:
```python
{code}
```

Generate a detailed specification including:
1. Method signature and parameters (exact names, types, defaults)
2. Return type and value
3. Internal logic and control flow
4. Variable names and their purposes
5. Comments and docstrings (exact text)
6. Error handling patterns
7. Edge cases and special conditions
8. Dependencies and imports
9. Side effects and state changes
10. Class attribute usage
11. Method calls and their context
12. Any other details needed for exact recreation

Format the specification as structured JSON with clear, detailed descriptions.
{context}
""",
            
            'algorithm_function': """
Analyze the following algorithm implementation and generate a comprehensive specification that would allow an LLM to regenerate the exact same code with 95%+ similarity.

Algorithm function to analyze:
```python
{code}
```

Generate a detailed specification including:
1. Function signature and parameters (exact names, types, defaults)
2. Return type and value
3. Algorithm steps and logic flow (detailed)
4. Variable names and their purposes
5. Comments and docstrings (exact text)
6. Error handling patterns
7. Edge cases and special conditions
8. Dependencies and imports
9. Side effects and state changes
10. Algorithm complexity and performance characteristics
11. Data structures and their usage
12. Any other details needed for exact recreation

Pay special attention to:
- Algorithm logic and step-by-step process
- Data structure usage and manipulation
- Performance considerations
- Edge cases and boundary conditions

Format the specification as structured JSON with clear, detailed descriptions.
{context}
"""
        }
    
    def _generate_strategy_prompt(self, strategy: str, code: str, context: Dict[str, Any]) -> str:
        """Generate prompt using selected strategy"""
        template = self.prompt_templates.get(strategy, self.prompt_templates['simple_function'])
        
        # Format context with special handling for class methods
        context_str = ""
        if context:
            # Add class context if this is a class method
            if context.get('is_class_method') and context.get('class_context'):
                class_ctx = context['class_context']
                context_str += f"\n\nClass Context (IMPORTANT - This method belongs to a class):"
                context_str += f"\n- Class Name: {class_ctx.get('class_name', 'Unknown')}"
                context_str += f"\n- Class Attributes (instance variables): {class_ctx.get('class_attributes', [])}"
                context_str += f"\n- Other Methods in Class: {class_ctx.get('other_methods', [])}"
                if class_ctx.get('class_docstring'):
                    context_str += f"\n- Class Purpose: {class_ctx.get('class_docstring')}"
                context_str += "\n\nREMEMBER: This is a class method. It MUST:"
                context_str += "\n1. Use 'self' as the first parameter"
                context_str += f"\n2. Access class attributes via self.attribute (available: {class_ctx.get('class_attributes', [])})"
                context_str += "\n3. Match the exact instance variable usage patterns"
                context_str += "\n4. Maintain state consistency with the class"
            
            # Add other context
            filtered_context = {k: v for k, v in context.items() 
                              if k not in ['is_class_method', 'class_context', 'code_analysis']}
            if filtered_context:
                context_str += f"\n\nAdditional context:\n{json.dumps(filtered_context, indent=2, default=str)}"
        
        return template.format(code=code, context=context_str)
    
    def _analyze_feedback_patterns(self, feedback: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """Analyze feedback patterns to identify improvement areas"""
        analysis = {
            'similarity_gaps': feedback.get('gaps', []),
            'metrics': feedback.get('metrics', {}),
            'priority_issues': feedback.get('priority', []),
            'suggestions': feedback.get('suggestions', []),
            'iteration': iteration,
            'improvement_trend': self._calculate_improvement_trend(feedback)
        }
        
        # Categorize issues by type
        analysis['issue_categories'] = self._categorize_issues(analysis['similarity_gaps'])
        
        # Identify patterns
        analysis['patterns'] = self._identify_patterns(analysis['similarity_gaps'])
        
        return analysis
    
    def _generate_improvement_strategies(self, feedback_analysis: Dict[str, Any], iteration: int) -> List[str]:
        """Generate improvement strategies based on feedback analysis"""
        strategies = []
        
        # Base strategies for all iterations
        strategies.append("Be more specific and detailed in the specification")
        strategies.append("Include exact variable names and their usage patterns")
        strategies.append("Specify exact control flow and logic structure")
        
        # Iteration-specific strategies
        if iteration > 1:
            strategies.append("Focus on the specific gaps identified in previous iterations")
            strategies.append("Provide more detailed examples and edge cases")
        
        if iteration > 2:
            strategies.append("Use more precise language and avoid ambiguity")
            strategies.append("Include exact formatting and style requirements")
        
        # Category-specific strategies
        categories = feedback_analysis.get('issue_categories', {})
        
        if categories.get('structural'):
            strategies.append("Provide detailed structural information including nesting and organization")
        
        if categories.get('semantic'):
            strategies.append("Include more semantic context and meaning")
        
        if categories.get('functional'):
            strategies.append("Specify exact functional behavior and logic flow")
        
        if categories.get('imports'):
            strategies.append("Include all necessary imports and dependencies")
        
        if categories.get('comments'):
            strategies.append("Include exact comments and docstrings")
        
        if categories.get('variables'):
            strategies.append("Specify exact variable names and their purposes")
        
        if categories.get('control_flow'):
            strategies.append("Provide detailed control flow information")
        
        # Pattern-specific strategies
        patterns = feedback_analysis.get('patterns', {})
        
        if patterns.get('recursion'):
            strategies.append("Focus on base case and recursive case conditions")
        
        if patterns.get('loops'):
            strategies.append("Specify exact loop conditions and iteration patterns")
        
        if patterns.get('conditionals'):
            strategies.append("Detail all conditional logic and branching")
        
        return strategies
    
    def _apply_improvements(self, original_prompt: str, improvements: List[str], iteration: int) -> str:
        """Apply improvements to the original prompt"""
        improved_prompt = original_prompt
        
        # Add improvement instructions
        if improvements:
            improvement_text = "\n\nPay special attention to:\n"
            for i, improvement in enumerate(improvements, 1):
                improvement_text += f"{i}. {improvement}\n"
            
            improved_prompt += improvement_text
        
        # Add iteration-specific instructions
        if iteration > 1:
            improved_prompt += f"\n\nThis is iteration {iteration}. Focus on addressing the specific issues identified in previous iterations."
        
        if iteration > 3:
            improved_prompt += "\n\nUse the most precise language possible and include every detail needed for exact recreation."
        
        return improved_prompt
    
    def _categorize_issues(self, gaps: List[str]) -> Dict[str, List[str]]:
        """Categorize similarity gaps by type"""
        categories = {
            'structural': [],
            'semantic': [],
            'functional': [],
            'textual': [],
            'imports': [],
            'comments': [],
            'variables': [],
            'control_flow': []
        }
        
        for gap in gaps:
            gap_lower = gap.lower()
            if any(keyword in gap_lower for keyword in ['structural', 'organization', 'structure']):
                categories['structural'].append(gap)
            elif any(keyword in gap_lower for keyword in ['semantic', 'meaning', 'context']):
                categories['semantic'].append(gap)
            elif any(keyword in gap_lower for keyword in ['functional', 'behavior', 'logic']):
                categories['functional'].append(gap)
            elif any(keyword in gap_lower for keyword in ['import', 'dependency']):
                categories['imports'].append(gap)
            elif any(keyword in gap_lower for keyword in ['comment', 'docstring', 'documentation']):
                categories['comments'].append(gap)
            elif any(keyword in gap_lower for keyword in ['variable', 'name']):
                categories['variables'].append(gap)
            elif any(keyword in gap_lower for keyword in ['control', 'flow', 'loop', 'condition']):
                categories['control_flow'].append(gap)
            else:
                categories['textual'].append(gap)
        
        return categories
    
    def _identify_patterns(self, gaps: List[str]) -> Dict[str, bool]:
        """Identify patterns in similarity gaps"""
        patterns = {
            'recursion': False,
            'loops': False,
            'conditionals': False,
            'error_handling': False,
            'data_structures': False
        }
        
        for gap in gaps:
            gap_lower = gap.lower()
            if any(keyword in gap_lower for keyword in ['recursive', 'recursion']):
                patterns['recursion'] = True
            elif any(keyword in gap_lower for keyword in ['loop', 'iteration']):
                patterns['loops'] = True
            elif any(keyword in gap_lower for keyword in ['conditional', 'if', 'branch']):
                patterns['conditionals'] = True
            elif any(keyword in gap_lower for keyword in ['error', 'exception', 'try']):
                patterns['error_handling'] = True
            elif any(keyword in gap_lower for keyword in ['data', 'structure', 'list', 'dict']):
                patterns['data_structures'] = True
        
        return patterns
    
    def _calculate_improvement_trend(self, feedback: Dict[str, Any]) -> str:
        """Calculate improvement trend from feedback"""
        # Simplified trend calculation
        metrics = feedback.get('metrics', {})
        if not metrics:
            return 'unknown'
        
        # Check if similarity is improving
        current_similarity = metrics.get('overall_similarity', 0)
        if current_similarity > 0.9:
            return 'excellent'
        elif current_similarity > 0.8:
            return 'good'
        elif current_similarity > 0.7:
            return 'fair'
        else:
            return 'poor'


class ContextAnalyzer:
    """Analyzes code context for better prompt generation"""
    
    def analyze_code_advanced(self, code: str) -> Dict[str, Any]:
        """Analyze code characteristics in detail"""
        try:
            import ast
            tree = ast.parse(code)
            
            analysis = {
                'complexity': self._calculate_complexity(tree),
                'has_recursion': self._has_recursion(tree),
                'is_class_method': self._is_class_method(tree),
                'function_count': len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]),
                'class_count': len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]),
                'import_count': len([node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]),
                'has_error_handling': self._has_error_handling(tree),
                'has_comments': self._has_comments(code),
                'variable_count': self._count_variables(tree),
                'control_flow_complexity': self._calculate_control_flow_complexity(tree),
                'algorithm_patterns': self._detect_algorithm_patterns(tree),
                'design_patterns': self._detect_design_patterns(tree)
            }
            
            return analysis
            
        except:
            return {
                'complexity': 1,
                'has_recursion': False,
                'is_class_method': False,
                'function_count': 0,
                'class_count': 0,
                'import_count': 0,
                'has_error_handling': False,
                'has_comments': False,
                'variable_count': 0,
                'control_flow_complexity': 1,
                'algorithm_patterns': [],
                'design_patterns': []
            }
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _has_recursion(self, tree: ast.AST) -> bool:
        """Check if code has recursion"""
        function_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_names.add(node.name)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in function_names:
                    return True
        
        return False
    
    def _is_class_method(self, tree: ast.AST) -> bool:
        """Check if code is a class method"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if it's inside a class
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.ClassDef):
                        for item in parent.body:
                            if item == node:
                                return True
        return False
    
    def _has_error_handling(self, tree: ast.AST) -> bool:
        """Check if code has error handling"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                return True
        return False
    
    def _has_comments(self, code: str) -> bool:
        """Check if code has comments"""
        lines = code.split('\n')
        for line in lines:
            if line.strip().startswith('#'):
                return True
        return False
    
    def _count_variables(self, tree: ast.AST) -> int:
        """Count unique variables"""
        variables = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                variables.add(node.id)
        return len(variables)
    
    def _calculate_control_flow_complexity(self, tree: ast.AST) -> int:
        """Calculate control flow complexity"""
        complexity = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _detect_algorithm_patterns(self, tree: ast.AST) -> List[str]:
        """Detect algorithm patterns"""
        patterns = []
        
        # Check for common algorithm patterns
        if self._has_recursion(tree):
            patterns.append('recursive')
        
        # Check for loop patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                patterns.append('iterative')
                break
        
        return patterns
    
    def _detect_design_patterns(self, tree: ast.AST) -> List[str]:
        """Detect design patterns"""
        patterns = []
        
        # Check for common design patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if 'factory' in node.name.lower():
                    patterns.append('factory')
                elif 'singleton' in node.name.lower():
                    patterns.append('singleton')
        
        return patterns


class StrategySelector:
    """Selects appropriate strategies based on code analysis"""
    
    def select_strategy(self, code_analysis: Dict[str, Any], context: Dict[str, Any], iteration: int) -> str:
        """Select appropriate strategy based on analysis"""
        complexity = code_analysis.get('complexity', 1)
        has_recursion = code_analysis.get('has_recursion', False)
        is_class_method = code_analysis.get('is_class_method', False)
        algorithm_patterns = code_analysis.get('algorithm_patterns', [])
        
        # Strategy selection logic
        if has_recursion:
            return 'recursive_function'
        elif is_class_method:
            return 'class_method'
        elif 'recursive' in algorithm_patterns or complexity > 8:
            return 'algorithm_function'
        elif complexity > 5:
            return 'complex_function'
        else:
            return 'simple_function'


class PromptOptimizer:
    """Optimizes prompts for better results"""
    
    def optimize_prompt(self, prompt: str, code_analysis: Dict[str, Any], similarity_gaps: List[str] = None, iteration: int = 1) -> str:
        """Optimize prompt based on code analysis and gaps"""
        optimized_prompt = prompt
        
        # Add complexity-specific optimizations
        complexity = code_analysis.get('complexity', 1)
        if complexity > 5:
            optimized_prompt += "\n\nFor this complex function, provide extremely detailed specifications including all edge cases and error conditions."
        
        # Add recursion-specific optimizations
        if code_analysis.get('has_recursion', False):
            optimized_prompt += "\n\nFor this recursive function, clearly specify the base case and recursive case conditions."
        
        # Add error handling optimizations
        if code_analysis.get('has_error_handling', False):
            optimized_prompt += "\n\nInclude all error handling patterns and exception types."
        
        # Add algorithm-specific optimizations
        algorithm_patterns = code_analysis.get('algorithm_patterns', [])
        if algorithm_patterns:
            optimized_prompt += f"\n\nThis function implements {', '.join(algorithm_patterns)} patterns. Focus on the algorithmic logic and data structures used."
        
        # Add gap-specific optimizations
        if similarity_gaps:
            gap_instructions = self._generate_gap_instructions(similarity_gaps)
            optimized_prompt += f"\n\nAddress these specific issues:\n{gap_instructions}"
        
        # Add iteration-specific optimizations
        if iteration > 1:
            optimized_prompt += f"\n\nThis is iteration {iteration}. Focus on the specific gaps identified in previous iterations."
        
        return optimized_prompt
    
    def _generate_gap_instructions(self, gaps: List[str]) -> str:
        """Generate specific instructions for addressing gaps"""
        instructions = []
        
        for i, gap in enumerate(gaps, 1):
            if 'structural' in gap.lower():
                instructions.append(f"{i}. Provide detailed structural information")
            elif 'semantic' in gap.lower():
                instructions.append(f"{i}. Include more semantic context")
            elif 'functional' in gap.lower():
                instructions.append(f"{i}. Specify exact functional behavior")
            elif 'import' in gap.lower():
                instructions.append(f"{i}. Include all necessary imports")
            elif 'comment' in gap.lower():
                instructions.append(f"{i}. Include exact comments and docstrings")
            elif 'variable' in gap.lower():
                instructions.append(f"{i}. Specify exact variable names")
            else:
                instructions.append(f"{i}. Address: {gap}")
        
        return '\n'.join(instructions)
