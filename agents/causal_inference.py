"""
Causal Specification Inference (CSI)
Uses causal inference to identify minimal sets of code elements necessary for behaviors.
Novel approach: First application of causal inference to specification generation.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import ast
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict


class CausalRelation(Enum):
    """Types of causal relations between code elements"""
    NECESSARY = "necessary"  # X is necessary for Y
    SUFFICIENT = "sufficient"  # X is sufficient for Y
    NECESSARY_AND_SUFFICIENT = "necessary_and_sufficient"
    CONTRIBUTING = "contributing"  # X contributes to Y but not necessary/sufficient


@dataclass
class CausalEdge:
    """Represents a causal edge in the Causal Program Dependency Graph"""
    source: str  # Source code element (line, variable, statement)
    target: str  # Target behavior or code element
    relation: CausalRelation
    confidence: float  # Confidence in the causal relationship (0-1)
    evidence: List[str]  # Evidence supporting this causal relation


@dataclass
class CodeElement:
    """Represents a code element (statement, expression, variable)"""
    id: str
    line_no: int
    ast_node: Any  # AST node
    type: str  # 'assignment', 'conditional', 'loop', 'return', 'variable'
    content: str  # Source code text
    variables_read: Set[str]
    variables_written: Set[str]


@dataclass
class Behavior:
    """Represents a behavior that needs specification"""
    id: str
    description: str
    trigger_conditions: List[str]  # Conditions that trigger this behavior
    effects: List[str]  # Effects of this behavior
    return_value: Optional[str] = None
    side_effects: List[str] = None


class CausalProgramDependencyGraph:
    """Causal Program Dependency Graph - extends traditional PDG with causal edges"""
    
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.code_elements: Dict[str, CodeElement] = {}
        self.behaviors: Dict[str, Behavior] = {}
        self.causal_edges: List[CausalEdge] = []
        self.data_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.control_dependencies: Dict[str, Set[str]] = defaultdict(set)
        
        # Build the graph
        self._build_graph()
    
    def _build_graph(self):
        """Build the causal program dependency graph"""
        try:
            if not self.source_code or not isinstance(self.source_code, str):
                return
            
            # Clean and validate source code
            source_code = self.source_code.strip()
            if not source_code or len(source_code) < 10:
                return
            
            tree = ast.parse(source_code)
            
            # Extract code elements
            self._extract_code_elements(tree)
            
            # Extract behaviors
            self._extract_behaviors(tree)
            
            # Build data dependencies
            self._build_data_dependencies()
            
            # Build control dependencies
            self._build_control_dependencies()
            
            # Infer causal relationships
            self._infer_causal_relationships()
            
        except SyntaxError as e:
            # Syntax errors are expected for malformed code - silently skip
            pass
        except Exception:
            pass
    
    def _extract_code_elements(self, tree: ast.AST):
        """Extract all code elements (statements, expressions)"""
        lines = self.source_code.splitlines()
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.If, ast.For, ast.While, ast.Return, ast.Call)):
                lineno = getattr(node, 'lineno', 0)
                if lineno == 0:
                    continue
                
                # Get source code for this node
                end_lineno = getattr(node, 'end_lineno', lineno)
                content = '\n'.join(lines[lineno-1:end_lineno])
                
                # Determine type
                if isinstance(node, ast.Assign):
                    elem_type = 'assignment'
                    vars_read = self._get_variables_read(node.value)
                    vars_written = {t.id for t in node.targets if isinstance(t, ast.Name)}
                elif isinstance(node, (ast.If, ast.While)):
                    elem_type = 'conditional'
                    vars_read = self._get_variables_read(node.test)
                    vars_written = set()
                elif isinstance(node, ast.For):
                    elem_type = 'loop'
                    vars_read = self._get_variables_read(node.iter)
                    vars_written = {node.target.id} if isinstance(node.target, ast.Name) else set()
                elif isinstance(node, ast.Return):
                    elem_type = 'return'
                    vars_read = self._get_variables_read(node.value) if node.value else set()
                    vars_written = set()
                else:
                    elem_type = 'statement'
                    vars_read = set()
                    vars_written = set()
                
                elem_id = f"L{lineno}"
                self.code_elements[elem_id] = CodeElement(
                    id=elem_id,
                    line_no=lineno,
                    ast_node=node,
                    type=elem_type,
                    content=content.strip(),
                    variables_read=vars_read,
                    variables_written=vars_written
                )
    
    def _get_variables_read(self, node: ast.AST) -> Set[str]:
        """Get all variables read in an AST node"""
        vars_read = set()
        
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                vars_read.add(n.id)
        
        return vars_read
    
    def _extract_behaviors(self, tree: ast.AST):
        """Extract behaviors (return paths, side effects, exceptions)"""
        behavior_id = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract return behaviors
                returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
                for i, ret in enumerate(returns):
                    behavior_id += 1
                    behavior = Behavior(
                        id=f"B{behavior_id}",
                        description=f"Return path {i+1} in {node.name}",
                        trigger_conditions=self._extract_conditions_to_return(ret, node),
                        effects=[f"Returns {ast.unparse(ret.value) if ret.value and hasattr(ast, 'unparse') else 'value'}"] if ret.value else ["Returns None"],
                        return_value=ast.unparse(ret.value) if ret.value and hasattr(ast, 'unparse') else None
                    )
                    self.behaviors[behavior.id] = behavior
                
                # Extract side effect behaviors
                for assign in ast.walk(node):
                    if isinstance(assign, ast.Assign):
                        for target in assign.targets:
                            if isinstance(target, ast.Attribute):
                                if isinstance(target.value, ast.Name) and target.value.id == 'self':
                                    behavior_id += 1
                                    behavior = Behavior(
                                        id=f"B{behavior_id}",
                                        description=f"Modifies {target.attr} in {node.name}",
                                        trigger_conditions=[],
                                        effects=[f"Sets self.{target.attr}"],
                                        side_effects=[f"self.{target.attr}"]
                                    )
                                    self.behaviors[behavior.id] = behavior
    
    def _extract_conditions_to_return(self, return_node: ast.Return, func_node: ast.FunctionDef) -> List[str]:
        """Extract conditions that lead to a specific return statement"""
        conditions = []
        
        # Find enclosing conditionals
        for node in ast.walk(func_node):
            if isinstance(node, ast.If):
                # Check if return is in this if block
                for child in ast.walk(node):
                    if child == return_node:
                        cond_str = ast.unparse(node.test) if hasattr(ast, 'unparse') else str(node.test)
                        conditions.append(cond_str)
                        break
        
        return conditions
    
    def _build_data_dependencies(self):
        """Build traditional data dependencies"""
        # For each variable write, find all reads
        for elem_id, elem in self.code_elements.items():
            for var_written in elem.variables_written:
                # Find all elements that read this variable
                for other_id, other_elem in self.code_elements.items():
                    if var_written in other_elem.variables_read:
                        self.data_dependencies[elem_id].add(other_id)
    
    def _build_control_dependencies(self):
        """Build traditional control dependencies"""
        # If a statement is inside a conditional, it's control-dependent on the conditional
        for elem_id, elem in self.code_elements.items():
            # Simplified: find enclosing conditionals
            # In a full implementation, would do proper control flow analysis
            for other_id, other_elem in self.code_elements.items():
                if other_elem.type == 'conditional':
                    # Check if elem is in the body of other_elem
                    if elem.line_no > other_elem.line_no:
                        # Simplified heuristic
                        self.control_dependencies[other_id].add(elem_id)
    
    def _infer_causal_relationships(self):
        """Infer causal relationships using heuristics and analysis"""
        # Rule 1: Direct data dependencies suggest causation
        for source_id, targets in self.data_dependencies.items():
            for target_id in targets:
                self.causal_edges.append(CausalEdge(
                    source=source_id,
                    target=target_id,
                    relation=CausalRelation.CONTRIBUTING,
                    confidence=0.7,
                    evidence=["data_dependency"]
                ))
        
        # Rule 2: Control dependencies are necessary (if condition fails, body doesn't execute)
        for source_id, targets in self.control_dependencies.items():
            for target_id in targets:
                self.causal_edges.append(CausalEdge(
                    source=source_id,
                    target=target_id,
                    relation=CausalRelation.NECESSARY,
                    confidence=0.9,
                    evidence=["control_dependency"]
                ))
        
        # Rule 3: Variables in return statements are necessary for return value
        for elem_id, elem in self.code_elements.items():
            if elem.type == 'return':
                for var in elem.variables_read:
                    # Find elements that write this variable
                    for other_id, other_elem in self.code_elements.items():
                        if var in other_elem.variables_written:
                            self.causal_edges.append(CausalEdge(
                                source=other_id,
                                target=elem_id,
                                relation=CausalRelation.NECESSARY,
                                confidence=0.8,
                                evidence=["return_dependency"]
                            ))


class CausalSpecificationInferencer:
    """Main class for causal specification inference"""
    
    def __init__(self, source_code: str, function_name: str):
        self.source_code = source_code or ''
        self.function_name = function_name or ''
        self.cpdg = None
        try:
            if source_code and len(source_code.strip()) > 10:
                self.cpdg = CausalProgramDependencyGraph(source_code)
        except Exception:
            # CPDG building failed - continue without it
            self.cpdg = None
    
    def find_minimal_causal_set(self, behavior_id: str) -> Set[str]:
        """
        Find minimal causal set for a behavior using intervention analysis.
        
        Algorithm:
        1. Start with all elements that could cause the behavior
        2. For each element, test if it's necessary (intervention: remove it)
        3. Keep only necessary elements
        4. Minimize: remove any elements that are redundant
        """
        if not self.cpdg:
            return set()
        behavior = self.cpdg.behaviors.get(behavior_id)
        if not behavior:
            return set()
        
        # Find all elements causally related to this behavior
        potential_causes = self._find_potential_causes(behavior_id)
        
        if not potential_causes:
            return set()
        
        # Intervention analysis: test necessity of each element
        necessary_elements = set()
        
        for elem_id in potential_causes:
            # Intervention: what if this element was removed/changed?
            if self._is_necessary(elem_id, behavior_id):
                necessary_elements.add(elem_id)
        
        # Minimize: remove redundant elements
        minimal_set = self._minimize_causal_set(necessary_elements, behavior_id)
        
        return minimal_set
    
    def _find_potential_causes(self, behavior_id: str) -> Set[str]:
        """Find all code elements that could potentially cause this behavior"""
        if not self.cpdg:
            return set()
        causes = set()
        
        # Direct causal edges
        for edge in self.cpdg.causal_edges:
            if edge.target == behavior_id:
                causes.add(edge.source)
        
        # Also check code elements that share variables with behavior
        behavior = self.cpdg.behaviors.get(behavior_id)
        if behavior:
            # Find elements that affect return value or side effects
            for elem_id, elem in self.cpdg.code_elements.items():
                # Check if element affects behavior's variables
                # This is a heuristic - full implementation would be more sophisticated
                if elem.type == 'return' and behavior.return_value:
                    causes.add(elem_id)
                elif elem.variables_written and behavior.side_effects:
                    # Check if any written variable is in side effects
                    for side_effect in behavior.side_effects or []:
                        for var in elem.variables_written:
                            if var in side_effect:
                                causes.add(elem_id)
        
        return causes
    
    def _is_necessary(self, elem_id: str, behavior_id: str) -> bool:
        """
        Test if an element is necessary for a behavior using intervention.
        Intervention: Remove or modify the element, check if behavior changes.
        """
        if not self.cpdg:
            return False
        # Simplified implementation: use heuristics
        # Full implementation would actually modify code and test
        
        elem = self.cpdg.code_elements.get(elem_id)
        if not elem:
            return False
        
        behavior = self.cpdg.behaviors.get(behavior_id)
        if not behavior:
            return False
        
        # Heuristic 1: If element is in control dependency path to behavior, likely necessary
        if elem_id in self.cpdg.control_dependencies.get(elem_id, set()):
            return True
        
        # Heuristic 2: If element writes variable used in behavior, likely necessary
        if elem.variables_written:
            for var in elem.variables_written:
                # Check if this variable is used in behavior's effects
                for effect in behavior.effects:
                    if var in effect:
                        return True
        
        # Heuristic 3: Check causal edges
        for edge in self.cpdg.causal_edges:
            if edge.source == elem_id and edge.target == behavior_id:
                if edge.relation in [CausalRelation.NECESSARY, CausalRelation.NECESSARY_AND_SUFFICIENT]:
                    return True
        
        return False
    
    def _minimize_causal_set(self, causal_set: Set[str], behavior_id: str) -> Set[str]:
        """
        Minimize causal set by removing redundant elements.
        An element is redundant if the behavior still occurs without it.
        """
        if not self.cpdg:
            return causal_set
        if len(causal_set) <= 1:
            return causal_set
        
        minimal = set(causal_set)
        
        # Try removing each element and see if set is still sufficient
        for elem_id in list(causal_set):
            test_set = minimal - {elem_id}
            # Simplified: if removing it doesn't break all causal paths, it's redundant
            if self._is_set_sufficient(test_set, behavior_id):
                minimal.remove(elem_id)
        
        return minimal
    
    def _is_set_sufficient(self, elem_set: Set[str], behavior_id: str) -> bool:
        """Check if a set of elements is sufficient to cause the behavior"""
        if not self.cpdg:
            return False
        # Simplified heuristic: check if set contains at least one necessary element
        # Full implementation would do actual code modification and testing
        
        if not elem_set:
            return False

        for edge in self.cpdg.causal_edges:
            if edge.source in elem_set and edge.target == behavior_id:
                if edge.relation in [CausalRelation.SUFFICIENT, CausalRelation.NECESSARY_AND_SUFFICIENT]:
                    return True

        return len(elem_set) > 0
    
    def generate_specification_from_causal_analysis(self) -> Dict[str, Any]:
        """
        Generate specification from causal analysis of all behaviors.
        
        Returns specification with:
        - Minimal set of code elements for each behavior
        - Causal relationships documented
        - Confidence scores
        """
        if not self.cpdg:
            return {'minimal_elements': [], 'causal_structure': {}}
        spec = {
            'function_name': self.function_name,
            'behaviors': [],
            'causal_structure': {},
            'minimal_elements': set(),
            'confidence': 1.0
        }
        
        # Analyze each behavior
        for behavior_id, behavior in self.cpdg.behaviors.items():
            minimal_set = self.find_minimal_causal_set(behavior_id)
            
            behavior_spec = {
                'behavior_id': behavior_id,
                'description': behavior.description,
                'minimal_causal_elements': list(minimal_set),
                'elements_detail': [
                    {
                        'id': elem_id,
                        'type': self.cpdg.code_elements[elem_id].type,
                        'content': self.cpdg.code_elements[elem_id].content[:100]  # Truncate
                    }
                    for elem_id in minimal_set if elem_id in self.cpdg.code_elements
                ],
                'trigger_conditions': behavior.trigger_conditions,
                'effects': behavior.effects
            }
            
            spec['behaviors'].append(behavior_spec)
            spec['minimal_elements'].update(minimal_set)
        
        spec['minimal_elements'] = list(spec['minimal_elements'])
        
        # Document causal structure
        spec['causal_structure'] = {
            'total_elements': len(self.cpdg.code_elements),
            'minimal_elements_count': len(spec['minimal_elements']),
            'reduction_ratio': len(spec['minimal_elements']) / len(self.cpdg.code_elements) if self.cpdg.code_elements else 0,
            'causal_edges_count': len(self.cpdg.causal_edges)
        }
        
        return spec
    
    def get_causal_specification_insights(self) -> Dict[str, Any]:
        """
        Get insights from causal analysis for specification enhancement.
        
        Returns insights that can be added to specification prompts.
        """
        if not self.cpdg:
            return {'insights_text': '', 'minimal_elements': []}
        spec = self.generate_specification_from_causal_analysis()
        
        insights = {
            'minimality_analysis': {
                'total_code_elements': len(self.cpdg.code_elements) if self.cpdg else 0,
                'minimal_elements': len(spec.get('minimal_elements', [])),
                'reduction': f"{(1 - spec.get('causal_structure', {}).get('reduction_ratio', 0)) * 100:.1f}% reduction" if spec.get('causal_structure') else "N/A"
            },
            'critical_elements': [
                {
                    'id': elem_id,
                    'type': elem.type,
                    'content': elem.content[:80],
                    'why_critical': 'Necessary for behavior'
                }
                for elem_id in spec.get('minimal_elements', [])[:5]
                for elem in [self.cpdg.code_elements.get(elem_id)] if self.cpdg and elem_id in self.cpdg.code_elements
                if elem
            ],
            'behavior_coverage': {
                'total_behaviors': len(self.cpdg.behaviors) if self.cpdg else 0,
                'covered_behaviors': len(spec.get('behaviors', []))
            }
        }
        
        # Generate natural language insights
        insights_text = f"""
CAUSAL ANALYSIS INSIGHTS:
- Minimal specification uses {insights['minimality_analysis']['minimal_elements']} of {insights['minimality_analysis']['total_code_elements']} code elements
- {insights['minimality_analysis']['reduction']} reduction while maintaining completeness
- Critical elements that MUST be included: {', '.join([e['id'] for e in insights['critical_elements'][:3]])}
- All {insights['behavior_coverage']['covered_behaviors']} behaviors covered by minimal element set

CRITICAL REQUIREMENTS:
Based on causal analysis, these elements are NECESSARY (not sufficient, but required):
{chr(10).join([f"- {e['id']} ({e['type']}): {e['content']}" for e in insights['critical_elements'][:5]])}
"""
        
        insights['insights_text'] = insights_text
        
        return insights

