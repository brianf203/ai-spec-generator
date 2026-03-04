"""
Program Slicing Analyzer for Advanced Code Decomposition
Uses program dependency graphs and slicing to identify semantically related code segments
"""

import ast
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, deque
import textwrap


class ProgramSlicingAnalyzer:
    """
    Advanced program slicing analyzer that identifies code segments
    affecting specific variables or outputs using program dependency graphs.
    
    More sophisticated than simple path enumeration because it:
    1. Understands data dependencies, not just control flow
    2. Identifies which statements actually affect outputs
    3. Produces smaller, more focused code segments
    """
    
    def __init__(self):
        self.pdg = None  # Program Dependency Graph
        self.def_use_chains = None  # Definition-Use chains
        self.parent_map: Dict[ast.AST, ast.AST] = {}
        self.source_lines: List[str] = []
    
    def analyze_with_slicing(self, code: str) -> Dict[str, Any]:
        """Analyze function using program slicing"""
        try:
            tree = ast.parse(code)
            self.source_lines = textwrap.dedent(code).splitlines()
            func_node = self._find_function(tree)
            if not func_node:
                return {'slices': [], 'pdg': {}}
            
            # Build Program Dependency Graph
            self.pdg = self._build_pdg(func_node)
            self.parent_map = self._build_parent_map(func_node)
            
            # Build definition-use chains
            self.def_use_chains = self._build_def_use_chains(func_node)
            
            # Identify slicing criteria (outputs, return values, side effects)
            slicing_criteria = self._identify_slicing_criteria(func_node)
            
            # Generate slices for each criterion
            slices = []
            for criterion in slicing_criteria:
                slice_statements = self._backward_slice(criterion, func_node)
                slice_info = self._analyze_slice(slice_statements, criterion, func_node)
                slices.append(slice_info)
            
            # Merge related slices (slices that share many statements)
            merged_slices = self._merge_related_slices(slices)
            
            # Identify independent loop structures
            loop_slices = self._identify_independent_loops(func_node, merged_slices)

            serialized_slices = [
                self._serialize_slice(slice_info, idx + 1)
                for idx, slice_info in enumerate(merged_slices)
            ]
            serialized_loops = [self._serialize_loop(loop) for loop in loop_slices]
            serialized_criteria = [self._sanitize_criterion(c) for c in slicing_criteria]
            
            return {
                'slices': serialized_slices,
                'loop_slices': serialized_loops,
                'slicing_criteria': serialized_criteria,
                'pdg_nodes': len(self.pdg),
                'complexity_reduction': self._calculate_complexity_reduction(func_node, merged_slices)
            }
        
        except Exception as e:
            return {'slices': [], 'error': str(e)}
    
    def _find_function(self, tree: ast.AST) -> Optional[ast.FunctionDef]:
        """Find function node in AST"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node
        return None
    
    def _build_pdg(self, func_node: ast.FunctionDef) -> Dict[ast.AST, Set[ast.AST]]:
        """
        Build Program Dependency Graph (PDG)
        
        PDG includes:
        - Data dependencies: variable definitions and uses
        - Control dependencies: control flow affecting execution
        """
        pdg = defaultdict(set)
        
        # Track variable definitions and uses
        var_defs = {}  # variable -> set of nodes that define it
        var_uses = defaultdict(set)  # variable -> set of nodes that use it
        
        def analyze_node(node, parent=None):
            """Analyze node for dependencies"""
            if isinstance(node, ast.Assign):
                # Variable definition
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_defs[target.id] = var_defs.get(target.id, set())
                        var_defs[target.id].add(node)
                
                # Add data dependency: assignment depends on RHS
                for child in ast.walk(node.value):
                    if isinstance(child, ast.Name):
                        if child.id in var_defs:
                            for def_node in var_defs[child.id]:
                                pdg[node].add(def_node)
            
            elif isinstance(node, ast.Name):
                # Variable use
                var_uses[node.id].add(node)
            
            elif isinstance(node, ast.If):
                # Control dependency: body depends on condition
                for stmt in node.body:
                    pdg[stmt].add(node.test)
            
            elif isinstance(node, ast.While):
                # Control dependency: body depends on condition
                for stmt in node.body:
                    pdg[stmt].add(node.test)
            
            elif isinstance(node, ast.For):
                # Control dependency: body depends on iteration
                for stmt in node.body:
                    pdg[stmt].add(node.iter)
            
            # Recursively analyze children
            for child in ast.iter_child_nodes(node):
                analyze_node(child, node)
        
        analyze_node(func_node)
        
        # Add data dependencies from uses to definitions
        for var, use_nodes in var_uses.items():
            if var in var_defs:
                for use_node in use_nodes:
                    for def_node in var_defs[var]:
                        pdg[use_node].add(def_node)
        
        return dict(pdg)
    
    def _build_def_use_chains(self, func_node: ast.FunctionDef) -> Dict[str, List[Tuple[ast.AST, ast.AST]]]:
        """
        Build definition-use chains: for each variable, track
        which definitions reach which uses
        """
        def_use = defaultdict(list)
        var_defs = {}  # variable -> list of (node, line) tuples
        
        def analyze_node(node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        line = getattr(node, 'lineno', 0)
                        var_defs[target.id] = var_defs.get(target.id, [])
                        var_defs[target.id].append((node, line))
            
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in var_defs:
                    for def_node, def_line in var_defs[node.id]:
                        line = getattr(node, 'lineno', 0)
                        def_use[node.id].append((def_node, node))
            
            for child in ast.iter_child_nodes(node):
                analyze_node(child)
        
        analyze_node(func_node)
        return dict(def_use)
    
    def _identify_slicing_criteria(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Identify slicing criteria: return values, side effects, outputs"""
        criteria = []
        
        # Find return statements
        returns = [n for n in ast.walk(func_node) if isinstance(n, ast.Return)]
        for ret in returns:
            if ret.value:
                criteria.append({
                    'type': 'return',
                    'node': ret.value,
                    'line': getattr(ret, 'lineno', None),
                    'description': f"Return value at line {getattr(ret, 'lineno', None)}",
                    'target_code': self._ast_to_str(ret.value)
                })
        
        # Find side effects (attribute assignments, calls)
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        criteria.append({
                            'type': 'side_effect',
                            'node': target,
                            'line': getattr(node, 'lineno', None),
                            'description': f"Attribute assignment: {self._ast_to_str(target)}",
                            'target_code': self._ast_to_str(target)
                        })
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    criteria.append({
                        'type': 'method_call',
                        'node': node,
                        'line': getattr(node, 'lineno', None),
                        'description': f"Method call: {self._ast_to_str(node.func)}",
                        'target_code': self._ast_to_str(node.func)
                    })
        
        return criteria
    
    def _backward_slice(self, criterion: Dict[str, Any], func_node: ast.FunctionDef) -> Set[ast.AST]:
        """
        Backward slice: find all statements that affect the criterion
        
        Algorithm:
        1. Start from criterion (e.g., return value)
        2. Follow dependencies backwards
        3. Collect all relevant statements
        """
        visited = set()
        slice_statements = set()
        worklist = deque()
        
        # Initialize worklist with criterion
        worklist.append(criterion.get('node'))
        
        while worklist:
            node = worklist.popleft()
            if node in visited:
                continue
            visited.add(node)
            
            # Add this node to slice
            slice_statements.add(node)
            
            # Follow dependencies backwards
            if node in self.pdg:
                for dep_node in self.pdg[node]:
                    if dep_node not in visited:
                        worklist.append(dep_node)
            
            # Also follow parent if it's a control structure
            # (we need to include the control structure itself)
            parent = self.parent_map.get(node) or self._find_parent(node, func_node)
            if parent and isinstance(parent, (ast.If, ast.While, ast.For)):
                if parent not in visited:
                    worklist.append(parent)
        
        return slice_statements
    
    def _find_parent(self, target: ast.AST, root: ast.AST) -> Optional[ast.AST]:
        """Find parent node of target in AST"""
        for parent in ast.walk(root):
            for child in ast.iter_child_nodes(parent):
                if child is target:
                    return parent
        return None
    
    def _analyze_slice(self, slice_statements: Set[ast.AST], criterion: Dict[str, Any], 
                      func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze a slice to extract useful information"""
        slice_lines = sorted([
            getattr(n, 'lineno', 0) for n in slice_statements 
            if hasattr(n, 'lineno') and n.lineno
        ])
        
        # Identify variables used in slice
        variables = set()
        for node in slice_statements:
            if isinstance(node, ast.Name):
                variables.add(node.id)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.add(target.id)
        
        # Identify control structures in slice
        control_structures = []
        for node in slice_statements:
            if isinstance(node, ast.If):
                control_structures.append('if')
            elif isinstance(node, ast.While):
                control_structures.append('while')
            elif isinstance(node, ast.For):
                control_structures.append('for')
        
        guard_conditions, guard_lines = self._collect_guard_context(slice_statements)
        
        return {
            'criterion': criterion,
            'statements': slice_statements,
            'statement_count': len(slice_statements),
            'lines': slice_lines,
            'line_range': (min(slice_lines) if slice_lines else 0, 
                          max(slice_lines) if slice_lines else 0),
            'variables': list(variables),
            'control_structures': control_structures,
            'complexity': len(control_structures),
            'guard_conditions': guard_conditions,
            'guard_lines': sorted(guard_lines)
        }
    
    def _merge_related_slices(self, slices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge slices that share many statements (overlap significantly)"""
        if not slices:
            return []
        
        merged = []
        used = set()
        
        for i, slice1 in enumerate(slices):
            if i in used:
                continue
            
            # Find slices that overlap significantly
            overlapping = [slice1]
            for j, slice2 in enumerate(slices[i+1:], i+1):
                if j in used:
                    continue
                
                overlap = self._calculate_overlap(slice1, slice2)
                if overlap > 0.5:  # More than 50% overlap
                    overlapping.append(slice2)
                    used.add(j)
            
            # Merge overlapping slices
            if len(overlapping) > 1:
                merged_slice = self._merge_slice_list(overlapping)
                merged.append(merged_slice)
            else:
                merged.append(slice1)
        
        return merged
    
    def _calculate_overlap(self, slice1: Dict[str, Any], slice2: Dict[str, Any]) -> float:
        """Calculate overlap ratio between two slices"""
        stmts1 = slice1['statements']
        stmts2 = slice2['statements']
        
        intersection = len(stmts1 & stmts2)
        union = len(stmts1 | stmts2)
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_slice_list(self, slices: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple slices into one"""
        merged_statements = set()
        merged_variables = set()
        merged_control = []
        all_criteria = []
        
        for slice_info in slices:
            merged_statements.update(slice_info['statements'])
            merged_variables.update(slice_info['variables'])
            merged_control.extend(slice_info['control_structures'])
            all_criteria.append(slice_info['criterion'])
        
        all_lines = []
        for slice_info in slices:
            all_lines.extend(slice_info['lines'])
        
        guard_conditions = []
        guard_lines = set()
        for slice_info in slices:
            guard_conditions.extend(slice_info.get('guard_conditions', []))
            guard_lines.update(slice_info.get('guard_lines', []))
        
        return {
            'criterion': {'type': 'merged', 'criteria': all_criteria},
            'statements': merged_statements,
            'statement_count': len(merged_statements),
            'lines': sorted(set(all_lines)),
            'line_range': (min(all_lines) if all_lines else 0, 
                          max(all_lines) if all_lines else 0),
            'variables': list(merged_variables),
            'control_structures': merged_control,
            'complexity': len(set(merged_control)),
            'merged_from': len(slices),
            'guard_conditions': guard_conditions,
            'guard_lines': sorted(guard_lines)
        }
    
    def _identify_independent_loops(self, func_node: ast.FunctionDef, 
                                   slices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify independent loop structures.
        
        Independent loops are loops that:
        - Don't share variables (except loop counters)
        - Don't have data dependencies between them
        - Can be analyzed separately
        """
        loops = [n for n in ast.walk(func_node) if isinstance(n, (ast.For, ast.While))]
        
        independent_loops = []
        for loop in loops:
            # Check if this loop is independent of others
            loop_vars = self._extract_loop_variables(loop)
            
            is_independent = True
            for other_loop in loops:
                if other_loop is loop:
                    continue
                other_vars = self._extract_loop_variables(other_loop)
                
                # Check for variable overlap (excluding loop counters)
                shared_vars = loop_vars & other_vars
                if shared_vars:
                    # Might not be independent
                    # Check if shared vars are just loop counters
                    if not all(self._is_loop_counter(v, loop, other_loop) for v in shared_vars):
                        is_independent = False
                        break
            
            if is_independent:
                independent_loops.append({
                    'loop': loop,
                    'line': getattr(loop, 'lineno', None),
                    'type': type(loop).__name__,
                    'variables': loop_vars
                })
        
        return independent_loops
    
    def _extract_loop_variables(self, loop_node) -> Set[str]:
        """Extract variables used in a loop"""
        variables = set()
        
        if isinstance(loop_node, ast.For):
            if isinstance(loop_node.target, ast.Name):
                variables.add(loop_node.target.id)
        
        for node in ast.walk(loop_node):
            if isinstance(node, ast.Name):
                variables.add(node.id)
        
        return variables
    
    def _is_loop_counter(self, var: str, loop1, loop2) -> bool:
        """Check if variable is just a loop counter"""
        # Simple heuristic: if variable is loop target, it's a counter
        if isinstance(loop1, ast.For) and isinstance(loop1.target, ast.Name):
            if loop1.target.id == var:
                return True
        if isinstance(loop2, ast.For) and isinstance(loop2.target, ast.Name):
            if loop2.target.id == var:
                return True
        return False
    
    def _calculate_complexity_reduction(self, func_node: ast.FunctionDef, 
                                       slices: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate how much complexity was reduced by slicing"""
        total_statements = len(list(ast.walk(func_node)))
        avg_slice_size = sum(s['statement_count'] for s in slices) / len(slices) if slices else 0
        
        reduction_ratio = 1.0 - (avg_slice_size / total_statements) if total_statements > 0 else 0.0
        
        return {
            'total_statements': total_statements,
            'num_slices': len(slices),
            'avg_slice_size': avg_slice_size,
            'reduction_ratio': reduction_ratio,
            'complexity_reduced': reduction_ratio > 0.3  # Significant reduction
        }
    
    def _ast_to_str(self, node: ast.AST) -> str:
        """Convert AST node to string"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._ast_to_str(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return f"{self._ast_to_str(node.func)}()"
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

    def _build_parent_map(self, root: ast.AST) -> Dict[ast.AST, ast.AST]:
        parent_map = {}
        for parent in ast.walk(root):
            for child in ast.iter_child_nodes(parent):
                parent_map[child] = parent
        return parent_map

    def _collect_guard_context(self, slice_statements: Set[ast.AST]) -> Tuple[List[str], Set[int]]:
        guards = []
        guard_lines = set()
        seen = set()
        for node in slice_statements:
            current = node
            while current in self.parent_map:
                parent = self.parent_map[current]
                if isinstance(parent, (ast.If, ast.While)):
                    key = (parent.lineno, ast.dump(parent.test))
                    if key not in seen:
                        seen.add(key)
                        guards.append(f"{type(parent).__name__.upper()} at line {getattr(parent, 'lineno', '?')}: {self._ast_to_str(parent.test)}")
                        guard_lines.add(getattr(parent, 'lineno', 0))
                current = parent
        return guards, guard_lines

    def _serialize_slice(self, slice_info: Dict[str, Any], index: int) -> Dict[str, Any]:
        criterion = slice_info.get('criterion', {})
        serialized_criterion = self._sanitize_criterion(criterion)
        statement_lines = slice_info.get('lines', [])
        snippets = [self._get_source_snippet(line) for line in statement_lines]
        return {
            'slice_id': f"slice_{index}",
            'criterion': serialized_criterion,
            'line_range': slice_info.get('line_range', (0, 0)),
            'statement_lines': statement_lines,
            'statement_snippets': [s for s in snippets if s],
            'variables': slice_info.get('variables', []),
            'control_structures': slice_info.get('control_structures', []),
            'complexity': slice_info.get('complexity', 0),
            'guard_conditions': slice_info.get('guard_conditions', []),
            'guard_lines': slice_info.get('guard_lines', []),
            'merged_from': slice_info.get('merged_from', 1)
        }

    def _sanitize_criterion(self, criterion: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(criterion, dict):
            return {}
        sanitized = {
            'type': criterion.get('type'),
            'line': criterion.get('line'),
            'description': criterion.get('description'),
            'target_code': criterion.get('target_code')
        }
        if criterion.get('type') == 'merged':
            sanitized['criteria'] = [
                self._sanitize_criterion(sub_criterion) for sub_criterion in criterion.get('criteria', [])
            ]
        return sanitized

    def _serialize_loop(self, loop_info: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'line': loop_info.get('line'),
            'type': loop_info.get('type'),
            'variables': sorted(loop_info.get('variables', []))
        }

    def _get_source_snippet(self, line_no: int) -> str:
        if not line_no or line_no - 1 >= len(self.source_lines) or line_no <= 0:
            return ""
        return self.source_lines[line_no - 1].strip()
