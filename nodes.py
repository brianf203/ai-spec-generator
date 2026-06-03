"""
Test generation, test execution, and behavioral validation
"""

import os
import ast
import json
import time
import re
import subprocess
import tempfile
import traceback
import importlib.util
import sys
import uuid
import textwrap
from contextlib import ExitStack
from typing import Dict, List, Any, Optional, Tuple, Set, Iterable
from collections import defaultdict
from functools import partial

from utils.claude_prompts import (
    SYSTEM_SPEC_JSON,
    SYSTEM_CODE_REGENERATION,
    SYSTEM_TEST_JSON,
    SYSTEM_FAILURE_DRIVEN_REFINEMENT,
    SYSTEM_SLICE_SPEC_JSON,
)
import numpy as np
from coverage import Coverage
from agents.advanced_analyzer import AdvancedCodeAnalyzer, SemanticSimilarityAnalyzer
from agents.smart_prompt_engine import SmartPromptEngine
from agents.program_slicing import ProgramSlicingAnalyzer
from agents.logical_deletion import LogicalDeletionPass
from agents.few_shot_prompter import FewShotPromptEnhancer
from agents.context_bundle import attach_bounded_context_to_functions


def _func_id_from_info(func_info: Dict[str, Any]) -> str:
    """Stable id for a function across pipeline stages (must match CodeAnalyzerNode keys)."""
    fid = func_info.get("func_id")
    if fid:
        return fid
    fp = func_info.get("file_path", "")
    qk = func_info.get("qualified_key") or func_info.get("function_name", "")
    return f"{fp}::{qk}"


def _coerce_to_float(value: Any, default: float = 0.0) -> float:
    """Safely coerce values into floats for similarity/coverage metrics."""
    try:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        return default
    except (TypeError, ValueError):
        return default


def compute_primary_similarity_metrics(
    structural_similarity: float,
    behavioral_test_similarity: float,
    total_test_cases: int,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Threshold primary metric: structural-only unless we have enough executed test cases.

    When the harness oracle is trustworthy (near-perfect agreement across many cases), optionally
    up-weight behavioral correctness so convergence does not require every cosmetic AST cue to
    match (``trusted_behavioral_oracle_blend``).
    """
    cfg = config or {}
    struct = max(0.0, min(1.0, _coerce_to_float(structural_similarity)))
    btest = max(0.0, min(1.0, _coerce_to_float(behavioral_test_similarity)))
    n_tests = max(0, int(total_test_cases or 0))
    min_needed = max(1, int(cfg.get("min_behavioral_cases", 3)))
    if n_tests < min_needed:
        return struct

    baseline = (struct + btest) / 2.0
    if not cfg.get("trusted_behavioral_oracle_blend", True):
        return baseline

    trust_floor = float(cfg.get("trusted_behavioral_agreement_floor", 0.999))
    alpha = max(0.0, min(1.0, float(cfg.get("behavioral_oracle_blend_weight", 0.72))))
    if btest + 1e-12 < trust_floor:
        return baseline

    oracle_mix = alpha * btest + (1.0 - alpha) * struct
    return min(1.0, max(baseline, oracle_mix))


REGEN_SPEC_ESSENTIAL_KEYS = frozenset(
    {
        "function_name",
        "signature",
        "return_type",
        "english_summary",
        "detailed_english_description",
        "variable_names",
        "control_flow",
        "error_handling",
        "detailed_step_by_step_logic",
        "detailed_variable_usage",
        "detailed_control_flow",
        "path_analysis",
        "slicing_analysis",
        "logical_deletion",
        "minimal_elements",
        "causal_analysis",
        "causal_insights",
        "causal_structure",
        "abstract_invariants",
        "max_nesting_depth",
        "user_stories",
        "success_criteria",
        "edge_cases",
        "class_context",
        "hybrid_code_additions",
        "hybrid_diff_summary",
        "structural_implementation_gaps",
        "bounded_context_bundle",
        "bounded_context_manifest",
        "hybrid_use_exact_code",
        "docstring",
    }
)




def build_regeneration_spec_json_blob(
    specification: Dict[str, Any], config: Dict[str, Any]
) -> str:
    """JSON projection of specification for regeneration; shrinks responsibly over budget."""
    from agents.context_bundle import truncate_utf8

    budget = max(4096, int(config.get("regeneration_spec_json_char_budget", 56_000)))
    dumped = json.dumps(specification, indent=2, default=str)
    if len(dumped) <= budget:
        return dumped

    pruned: Dict[str, Any] = {
        k: specification[k]
        for k in REGEN_SPEC_ESSENTIAL_KEYS
        if k in specification
    }

    for _round in range(14):
        out = json.dumps(pruned, indent=2, default=str)
        if len(out) <= budget:
            return out

        bcb = pruned.get("bounded_context_bundle")
        if isinstance(bcb, str) and len(bcb.encode("utf-8", errors="replace")) > 3072:
            pruned["bounded_context_bundle"] = (
                truncate_utf8(
                    bcb,
                    max(
                        2048,
                        int(len(bcb.encode("utf-8", errors="replace")) * 0.62),
                    ),
                )
                + "\n# [... bounded_context_bundle truncated ...]\n"
            )
            continue

        if pruned.pop("bounded_context_manifest", None) is not None:
            continue

        pruned["_budget_note"] = (
            "Additional non-essential spec fields stripped to fit regeneration prompt budget."
        )
        clipped = truncate_utf8(
            json.dumps(pruned, default=str),
            budget,
        )
        return (
            clipped
            + "\n# [... specification blob clipped to regeneration budget ...]\n"
        )

    return truncate_utf8(json.dumps(pruned, default=str), budget)


class BaseNode:
    """Base class for all workflow nodes"""

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
                
                for func_key, func_info in file_analysis.get('functions', {}).items():
                    # func_key is "name" or "ClassName.method_name" (unique per file)
                    func_name = func_key.split('.')[-1] if '.' in func_key else func_key
                    func_id = f"{file_path}::{func_key}"
                    
                    all_functions[func_id] = {
                        'func_id': func_id,
                        'qualified_key': func_key,
                        'file_path': file_path,
                        'function_name': func_name,
                        'source_code': func_info.get('source', ''),  # Use get with default
                        'complexity': func_info.get('complexity', 0),
                        'dependencies': func_info.get('calls', []),
                        'line_number': func_info.get('line_number', 0),
                        'imports': file_analysis.get('imports', []),
                        'docstring': func_info.get('docstring'),
                        'is_class_method': func_info.get('is_class_method', False),
                        'parent_class': func_info.get('parent_class')
                    }
                    
                    # Ensure source_code is not empty
                    if not all_functions[func_id]['source_code'] or not all_functions[func_id]['source_code'].strip():
                        print(f"        WARNING: Empty source code for {func_id}")
            
            except Exception as e:
                print(f"        WARNING: Error analyzing {file_path}: {e}")
                continue
        
        only_qn = self.config.get("only_qualified_names")
        if only_qn:
            allow = set(only_qn)
            before = len(all_functions)
            all_functions = {
                fid: info
                for fid, info in all_functions.items()
                if info.get("qualified_key") in allow
            }
            print(f"    only_qualified_names filter: {before} -> {len(all_functions)} functions")
            if not all_functions:
                raise ValueError(
                    "No functions left after only_qualified_names filter "
                    "(check names match AST keys like ClassName.method or top-level name)."
                )

        context['python_files'] = python_files
        attach_bounded_context_to_functions(
            project_path=project_path,
            analyzed_files=analyzed_files,
            all_functions=all_functions,
            python_files=python_files,
            config=self.config,
        )

        context['analyzed_files'] = analyzed_files
        context['all_functions'] = all_functions
        context['total_functions'] = len(all_functions)
        
        print(f"    Found {len(all_functions)} functions across {len(analyzed_files)} files")
        
        return context
    
    def _find_python_files(self, project_path: str) -> List[str]:
        """Find all Python files in the project, respecting include/exclude patterns.
        Skips test files, READMEs, and other non-source files per config."""
        import fnmatch
        python_files = []
        project_root = os.path.abspath(project_path)
        include = self.config.get('include_patterns', ['*.py'])
        exclude = self.config.get('exclude_patterns', ['*test*', 'tests/*', '__pycache__/*'])
        skip_dirs = {'__pycache__', '.git', '.pytest_cache', 'node_modules', '.venv', 'venv', 'tests'}

        for root, dirs, files in os.walk(project_root):
            dirs[:] = [d for d in dirs if d not in skip_dirs]

            for file in files:
                if not file.endswith('.py'):
                    continue
                full_path = os.path.join(root, file)
                try:
                    rel_path = os.path.relpath(full_path, project_root)
                except ValueError:
                    rel_path = full_path
                rel_forward = rel_path.replace(os.sep, '/')

                excluded = any(fnmatch.fnmatch(rel_forward, p) or fnmatch.fnmatch(file, p)
                               for p in exclude)
                if excluded:
                    continue
                included = any(fnmatch.fnmatch(rel_forward, p) or fnmatch.fnmatch(file, p)
                               for p in include)
                if included:
                    python_files.append(full_path)

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
        lines = content.splitlines()
        
        def extract_from_node(node, parent_class=None):
            """Recursively extract functions from nodes"""
            if isinstance(node, ast.FunctionDef):
                func_source = self._get_function_source(content, node, lines)
                if func_source and func_source.strip():  # Only add if source is not empty
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
                        'source': func_source,
                        'is_class_method': parent_class is not None,
                        'parent_class': parent_class
                    }
                    # Use qualified name for class methods to avoid conflicts
                    func_key = f"{parent_class}.{node.name}" if parent_class else node.name
                    functions[func_key] = func_info
        
        # Extract top-level functions and class methods
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                extract_from_node(node, parent_class=None)
            elif isinstance(node, ast.ClassDef):
                # Extract methods from classes
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        extract_from_node(item, parent_class=node.name)
        
        return functions
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements"""
        imports: List[str] = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        imports.append(f"import {alias.name} as {alias.asname}")
                    else:
                        imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    parts = []
                    for alias in node.names:
                        if alias.asname:
                            parts.append(f"{alias.name} as {alias.asname}")
                        else:
                            parts.append(alias.name)
                    import_line = f"from {node.module} import {', '.join(parts)}"
                    imports.append(import_line)
        
        # Preserve order while removing duplicates
        seen = set()
        unique_imports = []
        for imp in imports:
            if imp not in seen:
                unique_imports.append(imp)
                seen.add(imp)
        
        return unique_imports
    
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
    
    def _get_function_source(self, content: str, node: ast.FunctionDef, lines: List[str] = None) -> str:
        """Extract source code for a function"""
        if lines is None:
            lines = content.splitlines()
        
        start_line = node.lineno - 1
        if start_line < 0 or start_line >= len(lines):
            return ""
        
        # Use end_lineno if available, otherwise estimate
        if hasattr(node, 'end_lineno') and node.end_lineno:
            end_line = node.end_lineno
        else:
            # Fallback: find the end by looking for dedent
            end_line = start_line + 1
            indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
            for i in range(start_line + 1, len(lines)):
                line = lines[i]
                if not line.strip():  # Empty line, continue
                    continue
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= indent_level and line.strip():
                    break
                end_line = i + 1
        
        if end_line > len(lines):
            end_line = len(lines)
        
        # Extract and return the source
        source_lines = lines[start_line:end_line]
        if not source_lines:
            return ""
        
        return '\n'.join(source_lines)


class SpecificationGeneratorNode(BaseNode):
    """Generates specifications for functions"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from utils.call_llm import call_llm
        self.call_llm = call_llm
        self.few_shot_enhancer = FewShotPromptEnhancer()
        from agents.smart_prompt_engine import SmartPromptEngine
        from agents.advanced_analyzer import AdvancedCodeAnalyzer
        from agents.divide_conquer import DivideAndConquerAnalyzer, DeltaImprovementAlgorithm
        from agents.spec_refinement import SpecificationRefinementEngine
        from agents.example_driven import ExampleDrivenSpecEnhancer
        from agents.pattern_templates import PatternTemplateMatcher
        self.smart_prompt_engine = SmartPromptEngine()
        self.advanced_analyzer = AdvancedCodeAnalyzer()
        self.divide_conquer = DivideAndConquerAnalyzer()
        self.delta_improver = DeltaImprovementAlgorithm()
        self.refinement_engine = SpecificationRefinementEngine()
        self.example_enhancer = ExampleDrivenSpecEnhancer()
        self.pattern_matcher = PatternTemplateMatcher()
        self.program_slicer = ProgramSlicingAnalyzer()
        self.logical_deletion = LogicalDeletionPass()
        from agents.slice_by_slice_spec_generator import SliceBySliceSpecGenerator
        self.slice_by_slice_generator = SliceBySliceSpecGenerator(
            partial(call_llm, config=self.config, system=SYSTEM_SLICE_SPEC_JSON)
        )

    def _bounded_context_prompt_suffix(self, func_info: Dict[str, Any]) -> str:
        if not self.config.get("enable_context_bundle", True):
            return ""
        ctx = (func_info.get("context_bundle_text") or "").strip()
        if not ctx:
            return ""
        return (
            "\n\nREAD-ONLY BOUNDED DEPENDENCY CONTEXT (deterministic excerpts: same-module callees, "
            "class envelope, scoped k-hop; infer semantics — summarize in specification fields, "
            "do not paste this block verbatim into the JSON):\n\n"
            + ctx
        )

    def _attach_bounded_context_to_specification(
        self, specification: Dict[str, Any], func_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.config.get("enable_context_bundle", True):
            return specification
        from agents.context_bundle import truncate_utf8

        raw = (func_info.get("context_bundle_raw") or "").strip()
        if not raw:
            raw = (func_info.get("context_bundle_text") or "").strip()
        if not raw:
            return specification

        reg_cap = int(self.config.get("context_regen_bundle_chars", 24_384))
        specification["bounded_context_bundle"] = truncate_utf8(raw, reg_cap)

        mf = func_info.get("context_bundle_manifest") or []
        compact = []
        for m in mf[:120]:
            if not isinstance(m, dict):
                continue
            compact.append(
                {
                    "rule": m.get("rule"),
                    "key": m.get("key"),
                    "chars": m.get("chars"),
                    "sha256_preview": m.get("sha256_preview"),
                }
            )
        specification["bounded_context_manifest"] = compact
        return specification

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specifications for all functions"""
        print("    Generating specifications...")
        
        all_functions = context.get('all_functions', {})
        specifications = context.get('specifications', {})
        
        # In iteration > 1, identify functions that need refinement (low similarity)
        iteration = context.get('current_iteration', 1)
        functions_to_refine = set()
        if iteration > 1:
            similarity_results = context.get('similarity_results', {})
            for func_id, result in similarity_results.items():
                metrics = result.get('similarity_metrics', {})
                if metrics.get('primary_similarity', 1.0) < 0.85:
                    functions_to_refine.add(func_id)
            if functions_to_refine:
                print(f"    Found {len(functions_to_refine)} functions with similarity < 85% - will refine specs")
        
        for func_id, func_info in all_functions.items():
            # Skip if spec exists and is successful, UNLESS we need to refine it
            if func_id in specifications and specifications[func_id].get('success', False):
                if func_id not in functions_to_refine:
                    continue
                # Force regeneration for low-similarity functions in iteration > 1
                print(f"      Re-processing {func_info['function_name']} for refinement (similarity < 85%)...")
            
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
                        # DO NOT include original_code - specifications should be code-free
                        'imports': func_info.get('imports', []),
                        'docstring': func_info.get('docstring'),
                        'drift_issues': spec_result.get('drift_issues', [])
                    }
                    
                    # Store original code separately in context (NOT in specification)
                    if 'original_code' not in context:
                        context['original_code'] = {}
                    context['original_code'][func_id] = func_info['source_code']
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
        
        # Extract metrics for chunking decision
        control_flow = code_analysis.get('control_flow', {})
        complexity_metrics = code_analysis.get('complexity_metrics', {})
        data_flow = code_analysis.get('data_flow', {})
        
        # Add metrics needed for strategy selection
        code_analysis['cyclomatic_complexity'] = complexity_metrics.get('cyclomatic_complexity', complexity)
        code_analysis['num_branches'] = control_flow.get('if_statements', 0)
        code_analysis['num_loops'] = control_flow.get('loops', 0)
        
        # Count state changes (assignments to instance attributes for class methods)
        num_state_changes = 0
        if 'self' in source_code:
            assignments = data_flow.get('assignments', [])
            for assign in assignments:
                var_name = assign.get('variable', '')
                if var_name.startswith('self.'):
                    num_state_changes += 1
        code_analysis['num_state_changes'] = num_state_changes
        
        # Check if this is a class method
        is_class_method = 'self' in func_info.get('source_code', '')
        class_context = self._extract_class_context(func_info, context) if is_class_method else None
        
        drift_issues = []
        
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
        
        feedback = context.get('feedback_data', {}).get(_func_id_from_info(func_info), None)
        similarity_gaps = []
        iteration = context.get('current_iteration', 1)
        
        if feedback:
            similarity_gaps = feedback.get('gaps', [])
        
        # Incorporate runtime feedback from previous iterations
        func_id = _func_id_from_info(func_info)
        runtime_feedback = context.get('runtime_feedback', {}).get(func_id, [])
        if runtime_feedback:
            failure_summaries = []
            for fb_entry in runtime_feedback:
                if fb_entry.get('iteration', 0) < iteration:
                    failure_summaries.append(fb_entry.get('failures', ''))
            if failure_summaries:
                runtime_gap = f"Previous test failures (iterations {[fb.get('iteration') for fb in runtime_feedback if fb.get('iteration', 0) < iteration]}): " + "; ".join(failure_summaries)
                similarity_gaps.append(runtime_gap)
        
        # Also check if there are appended failures in existing spec
        existing_spec = context.get('specifications', {}).get(func_id, {})
        if existing_spec and existing_spec.get('specification', {}).get('appended_failures'):
            appended = existing_spec['specification']['appended_failures']
            if appended:
                failure_details = "; ".join([str(f) for f in appended[:3]])
                similarity_gaps.append(f"Test execution failures: {failure_details}")
        
        # Use advanced decomposition for very complex functions
        # First try program slicing (more sophisticated); disable for monolithic / ablation baselines
        enable_slicing = self.config.get('enable_program_slicing', True)
        if enable_slicing:
            slicing_analysis = self.program_slicer.analyze_with_slicing(source_code)
            use_slicing = (slicing_analysis.get('complexity_reduction', {}).get('complexity_reduced', False) and
                          len(slicing_analysis.get('slices', [])) > 0)
        else:
            slicing_analysis = {}
            use_slicing = False
        
        # Track if we used slice-by-slice generation
        used_slice_by_slice = False
        
        # Initialize deltas early to avoid scope issues
        deltas = []
        similarity_results = context.get('similarity_results', {})
        test_results = context.get('test_results', {})
        if func_id in similarity_results and func_id in test_results:
            metrics = similarity_results[func_id].get('similarity_metrics', {})
            test_data = test_results[func_id]
            deltas = self.delta_improver.identify_improvement_deltas(metrics, test_data)
        
        # If slicing is available, use slice-by-slice generation
        causal_minimal_elements = []
        if use_slicing and slicing_analysis.get('slices') and len(slicing_analysis.get('slices', [])) > 0:
            print(f"        Using slice-by-slice generation...")
            try:
                from agents.causal_inference import CausalSpecificationInferencer
                csi = CausalSpecificationInferencer(source_code, func_info.get('function_name', ''))
                causal_spec = csi.generate_specification_from_causal_analysis()
                causal_minimal_elements = causal_spec.get('minimal_elements', [])
                slice_result = self.slice_by_slice_generator.generate_spec_from_slices(
                    source_code,
                    slicing_analysis,
                    func_info.get('function_name', ''),
                    func_info,
                    causal_minimal_elements,
                    refinement_notes=similarity_gaps or None,
                )
                
                if slice_result['success']:
                    specification = slice_result['specification']
                    # Add slicing analysis metadata
                    specification['slicing_analysis'] = slicing_analysis
                    specification['specification_method'] = 'slice_by_slice'
                    used_slice_by_slice = True
                    # Continue with standard enhancement pipeline below
                else:
                    print(f"        WARNING: Slice-by-slice generation failed: {slice_result.get('error')}")
                    # Fall through to standard generation
                    use_slicing = False
            except Exception as e:
                print(f"        WARNING: Slice-by-slice generation exception: {e}")
                use_slicing = False
        
        # Standard specification generation (if not using slice-by-slice or if it failed)
        if not used_slice_by_slice:
            # Fall back to path enumeration if slicing doesn't help
            path_analysis = self.divide_conquer.analyze_function_paths(source_code)
            use_divide_conquer = not use_slicing and self.delta_improver.should_use_divide_conquer({
                'complexity': complexity,
                'cyclomatic_complexity': code_analysis.get('cyclomatic_complexity', complexity),
                'num_paths': path_analysis.get('complexity', 0),
                'branching_factor': path_analysis.get('branching_factor', 0)
            })
            
            # Deltas already initialized above
            
            # Detect code patterns and get template guidance
            matched_patterns = self.pattern_matcher.match_pattern(source_code, func_info.get('function_name', ''))
            pattern_guidance = self.pattern_matcher.get_template_guidance(matched_patterns)
            
            # Generate base prompt
            base_prompt = self.smart_prompt_engine.generate_adaptive_prompt(
                source_code, prompt_context, similarity_gaps, iteration
            )
            
            # Add pattern-based guidance
            if pattern_guidance:
                base_prompt += "\n\n" + pattern_guidance
            
            # Apply delta improvement if we have deltas
            if deltas and iteration > 1:
                prompt = self.delta_improver.generate_delta_focused_prompt(base_prompt, deltas, iteration)
            else:
                prompt = base_prompt
            
            # Add few-shot examples for specification generation
            prompt = self.few_shot_enhancer.add_spec_examples_to_prompt(prompt, source_code)
            
            # If using program slicing, enhance prompt with slice information
            if use_slicing and slicing_analysis.get('slices'):
                slices = slicing_analysis['slices']
                slice_info = f"\n\nPROGRAM SLICING ANALYSIS (Advanced Decomposition):\n"
                slice_info += f"This function has been decomposed into {len(slices)} semantic slices:\n"
                for i, slice_data in enumerate(slices[:5], 1):
                    criterion = slice_data.get('criterion', {})
                    criterion_desc = criterion.get('description', criterion.get('type', 'unknown'))
                    line_range = slice_data.get('line_range', (0, 0))
                    vars_count = len(slice_data.get('variables', []))
                    slice_info += f"  Slice {i}: {criterion_desc} (lines {line_range[0]}-{line_range[1]}, {vars_count} variables)\n"
                
                if slicing_analysis.get('loop_slices'):
                    loop_info = f"\nIndependent loop structures identified: {len(slicing_analysis['loop_slices'])}\n"
                    slice_info += loop_info
                
                reduction = slicing_analysis.get('complexity_reduction', {})
                if reduction.get('complexity_reduced'):
                    slice_info += f"\nComplexity reduced by {(reduction.get('reduction_ratio', 0) * 100):.1f}% through slicing.\n"
                
                slice_info += "\nGenerate specifications that cover ALL slices. Each user story should map to a specific slice and its dependencies."
                prompt += slice_info
            
            # Fall back to divide-and-conquer if slicing not used
            elif use_divide_conquer and path_analysis.get('paths'):
                paths = path_analysis['paths']
                path_info = f"\n\nDIVIDE-AND-CONQUER ANALYSIS:\n"
                path_info += f"This function has {len(paths)} distinct execution paths:\n"
                for i, path in enumerate(paths[:5], 1):
                    path_info += f"  Path {i}: {path.get('type', 'unknown')} - Conditions: {', '.join(path.get('conditions', [])[:2])}\n"
                path_info += "\nGenerate specifications that cover ALL paths. Each user story should map to a specific path."
                prompt += path_info

            prompt += self._bounded_context_prompt_suffix(func_info)

            try:
                response = self.call_llm(prompt, system=SYSTEM_SPEC_JSON)
                specification = self._parse_specification_response(response)
            except Exception as e:
                return {
                    'success': False,
                    'error': f'LLM call failed: {str(e)}'
                }
        
        # Common enhancement pipeline for both slice-by-slice and standard specs
        try:
            specification = self._ensure_english_summary(specification, func_info, code_analysis)
            specification = self._enhance_variable_names(specification, source_code)
            specification = self._ensure_spec_structure_fields(specification)
            
            # Validate specification before proceeding
            from agents.spec_validator import SpecificationValidator
            from agents.incremental_spec import IncrementalSpecBuilder
            spec_validator = SpecificationValidator()
            incremental_builder = IncrementalSpecBuilder()
            
            is_valid, validation_issues = spec_validator.validate(
                specification, source_code, func_info.get('function_name', '')
            )
            if validation_issues:
                validation_feedback = spec_validator.generate_validation_feedback(validation_issues)
                if not is_valid:
                    # Convert ValidationIssue objects to dicts for JSON serialization
                    specification['validation_errors'] = [
                        {
                            'severity': issue.severity,
                            'category': issue.category,
                            'message': issue.message,
                            'field': issue.field
                        }
                        for issue in validation_issues
                    ]
                    print(f"        WARNING: Specification validation found errors: {len([i for i in validation_issues if i.severity == 'error'])} error(s)")
            
            # Enhance with incremental spec building to fill gaps
            try:
                if source_code and len(source_code.strip()) > 10:
                    specification = incremental_builder.enhance_existing_spec(
                        specification, source_code, func_info.get('function_name', '')
                    )
            except Exception:
                # Continue without incremental enhancement if it fails
                pass
            
            # Add abstract interpretation for invariants
            from agents.abstract_interpretation import AbstractInterpreter
            abstract_interpreter = AbstractInterpreter()
            invariants = abstract_interpreter.infer_invariants(source_code)
            if invariants and any(invariants.values()):
                invariant_spec = abstract_interpreter.generate_invariant_specification(invariants)
                if invariant_spec:
                    specification['abstract_invariants'] = invariants
                    specification['invariant_description'] = invariant_spec
            
            # Add causal specification inference (NOVEL RESEARCH CONTRIBUTION)
            from agents.causal_inference import CausalSpecificationInferencer
            try:
                if source_code and len(source_code.strip()) > 10:
                    causal_inferencer = CausalSpecificationInferencer(
                        source_code, func_info.get('function_name', '')
                    )
                    if causal_inferencer.cpdg:  # Only use if CPDG was built successfully
                        causal_insights = causal_inferencer.get_causal_specification_insights()
                        causal_spec = causal_inferencer.generate_specification_from_causal_analysis()
                        
                        if causal_insights and causal_spec:
                            specification['causal_analysis'] = causal_spec
                            specification['causal_insights'] = causal_insights.get('insights_text', '')
                            # Add minimality information
                            specification['minimal_elements'] = causal_spec.get('minimal_elements', [])
                            specification['causal_structure'] = causal_spec.get('causal_structure', {})
            except Exception:
                # Silently continue without causal analysis if it fails
                pass
            
            # Apply logical deletion if we have slicing analysis (works for both paths)
            if slicing_analysis and slicing_analysis.get('slices'):
                logical_plan = self.logical_deletion.build_plan(
                    source_code,
                    slicing_analysis,
                    specification,
                    func_info.get('function_name', '')
                )
                if logical_plan:
                    specification['logical_deletion'] = logical_plan
            
            # Add Counter-Example Guided Specification Synthesis (CEGSS) - NOVEL
            from agents.cegss import CEGSSEngine
            try:
                iteration = context.get('current_iteration', 1)
                if iteration > 1:  # Apply CEGSS in later iterations
                    cegss_engine = CEGSSEngine(source_code, func_info.get('function_name', ''))
                    specification = cegss_engine.synthesize_specification(specification)
                    cegss_guidance = cegss_engine.generate_cegss_guidance(specification)
                    if cegss_guidance:
                        specification['cegss_guidance'] = cegss_guidance
            except Exception as e:
                if self.config.get('verbose'):
                    print(f"        WARNING: CEGSS failed: {e}")
                # Continue without CEGSS if it fails
            
            # Enhance with examples from code analysis
            code_examples = self.example_enhancer.extract_examples_from_code(source_code, func_info.get('function_name', ''))
            specification = self.example_enhancer.enhance_specification_with_examples(specification, code_examples)
            
            # Enhance specification with detailed documentation for complex functions
            from agents.enhanced_spec_generator import EnhancedSpecGenerator
            enhanced_gen = EnhancedSpecGenerator()
            specification = enhanced_gen.enhance_specification(
                specification, source_code, func_info.get('function_name', '')
            )
            
            drift_issues = self._detect_spec_drift(specification, func_info, code_analysis)
            if drift_issues:
                specification['drift_issues'] = drift_issues
                prompt_context['drift_issues'] = drift_issues
            
            # Apply iterative refinement if this is a later iteration
            iteration = context.get('current_iteration', 1)
            if iteration > 1:
                similarity_results = context.get('similarity_results', {})
                test_results = context.get('test_results', {})
                regenerated_code = context.get('regenerated_code', {})
                
                if func_id in similarity_results and func_id in regenerated_code:
                    metrics = similarity_results[func_id].get('similarity_metrics', {})
                    regen_code = regenerated_code[func_id].get('code', '')
                    
                    if regen_code and metrics.get('primary_similarity', 1.0) < 0.85:
                        # Analyze refinement opportunities
                        test_data = test_results.get(func_id, {}) if func_id in test_results else None
                        refinement_targets = self.refinement_engine.analyze_refinement_opportunities(
                            specification,
                            source_code,
                            regen_code,
                            metrics,
                            test_data
                        )
                        
                        # Apply refinements
                        if refinement_targets:
                            specification = self.refinement_engine.refine_specification(
                                specification,
                                refinement_targets,
                                source_code
                            )
                            
                            # Also enhance with test examples if available
                            if test_data:
                                test_examples = self.example_enhancer.extract_examples_from_tests(test_data)
                                specification = self.example_enhancer.enhance_specification_with_examples(
                                    specification,
                                    code_examples,
                                    test_examples
                                )
            
            # Add class context to specification if applicable
            if is_class_method and class_context:
                specification['class_context'] = class_context

            specification = self._attach_bounded_context_to_specification(
                specification, func_info
            )

            # Store decomposition analysis for potential future use
            if use_slicing:
                specification['slicing_analysis'] = slicing_analysis
                specification['decomposition_method'] = 'program_slicing'
            elif use_divide_conquer:
                specification['path_analysis'] = path_analysis
                specification['decomposition_method'] = 'path_enumeration'
            
            # Store complexity for adaptive iteration calculation
            if 'function_complexities' not in context:
                context['function_complexities'] = {}
            context['function_complexities'][func_id] = complexity
            
            return {
                'success': True,
                'specification': specification,
                'code_analysis': code_analysis,
                'drift_issues': drift_issues,
                'deltas': deltas
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
    
    def _enhance_variable_names(self, specification: Dict[str, Any], source_code: str) -> Dict[str, Any]:
        """Extract and enhance variable names from source code to ensure accuracy"""
        try:
            tree = ast.parse(textwrap.dedent(source_code))
            extracted_vars = {}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id
                            if var_name not in extracted_vars:
                                extracted_vars[var_name] = {
                                    'name': var_name,
                                    'purpose': f'Variable assigned at line {getattr(node, "lineno", "?")}',
                                    'line': getattr(node, 'lineno', None)
                                }
                elif isinstance(node, ast.FunctionDef):
                    for arg in node.args.args:
                        if arg.arg != 'self':
                            if arg.arg not in extracted_vars:
                                extracted_vars[arg.arg] = {
                                    'name': arg.arg,
                                    'purpose': f'Function parameter',
                                    'line': getattr(node, 'lineno', None)
                                }
            
            # Merge with existing variable_names if present
            existing_vars = specification.get('variable_names', [])
            if isinstance(existing_vars, dict):
                existing_vars = [{'name': k, 'purpose': v if isinstance(v, str) else v.get('purpose', '')} 
                                for k, v in existing_vars.items()]
            elif not isinstance(existing_vars, list):
                existing_vars = []
            
            # Create a map of existing vars
            existing_map = {v.get('name') if isinstance(v, dict) else str(v): v for v in existing_vars}
            
            # Merge: prefer existing with purpose, add extracted if missing
            merged_vars = []
            seen_names = set()
            
            for var in existing_vars:
                if isinstance(var, dict):
                    name = var.get('name', '')
                else:
                    name = str(var)
                    var = {'name': name}
                
                if name and name not in seen_names:
                    merged_vars.append(var)
                    seen_names.add(name)
            
            for name, var_info in extracted_vars.items():
                if name not in seen_names:
                    merged_vars.append(var_info)
                    seen_names.add(name)
            
            if merged_vars:
                specification['variable_names'] = merged_vars
            
            return specification
        except Exception:
            return specification
    
    def _ensure_english_summary(
        self,
        specification: Dict[str, Any],
        func_info: Dict[str, Any],
        code_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Guarantee an english_summary field is present and informative."""
        summary = specification.get('english_summary')
        if isinstance(summary, str) and summary.strip():
            specification['english_summary'] = summary.strip()
            return specification
        
        docstring = (func_info.get('docstring') or "").strip() if func_info else ""
        if docstring:
            specification['english_summary'] = docstring
            return specification
        
        specification['english_summary'] = self._build_summary_from_analysis(
            func_info,
            code_analysis,
            specification
        )
        return specification
    
    def _build_summary_from_analysis(
        self,
        func_info: Dict[str, Any],
        code_analysis: Dict[str, Any],
        specification: Dict[str, Any]
    ) -> str:
        """Create a fallback natural-language summary using available analysis."""
        func_name = func_info.get('function_name') if func_info else None
        class_ctx = specification.get('class_context', {})
        class_name = class_ctx.get('class_name') if isinstance(class_ctx, dict) else None
        
        if class_name:
            prefix = f"The method {func_name} of {class_name}" if func_name else f"This method of {class_name}"
        else:
            prefix = f"The function {func_name}" if func_name else "This function"
        
        summary_parts = [f"{prefix} implements the behavior described in the structured specification."]
        
        data_flow = (code_analysis or {}).get('data_flow', {}) if isinstance(code_analysis, dict) else {}
        variables = data_flow.get('variables') if isinstance(data_flow, dict) else None
        if variables:
            highlighted = ', '.join(list(dict.fromkeys(variables))[:3])
            if highlighted:
                summary_parts.append(f"It manipulates variables such as {highlighted}.")
        
        control_flow = (code_analysis or {}).get('control_flow', {}) if isinstance(code_analysis, dict) else {}
        if isinstance(control_flow, dict):
            branches = control_flow.get('if_statements', 0)
            loops = control_flow.get('loops', 0)
            try_blocks = control_flow.get('try_blocks', 0)
            descriptors = []
            if branches:
                descriptors.append("conditional branches")
            if loops:
                descriptors.append("iterative loops")
            if try_blocks:
                descriptors.append("error handling")
            if descriptors:
                summary_parts.append("It includes " + ", ".join(descriptors) + ".")
        
        side_effects = specification.get('side_effects')
        if side_effects:
            if isinstance(side_effects, str):
                side_effect_text = side_effects
            elif isinstance(side_effects, list):
                side_effect_text = ', '.join(side_effects[:3])
            else:
                side_effect_text = str(side_effects)
            if side_effect_text:
                summary_parts.append(f"Side effects include {side_effect_text}.")
        
        return_value = specification.get('return_value') or specification.get('returns')
        if return_value:
            if isinstance(return_value, str):
                summary_parts.append(f"It ultimately returns {return_value}.")
            elif isinstance(return_value, dict):
                return_desc = return_value.get('description') or return_value.get('value')
                if return_desc:
                    summary_parts.append(f"It ultimately returns {return_desc}.")
        
        if len(summary_parts) == 1:
            summary_parts.append("It should maintain the original state and side effects of the reference implementation.")
        
        return " ".join(part.strip() for part in summary_parts if part)
    
    def _ensure_spec_structure_fields(
        self,
        specification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ensure Spec Kit-style sections exist with consistent shapes."""
        if specification is None:
            specification = {}
        
        specification.setdefault('user_stories', [])
        specification.setdefault('success_criteria', [])
        specification.setdefault('test_matrix', [])
        
        edge_cases = specification.get('edge_cases')
        if edge_cases is None:
            specification['edge_cases'] = []
        elif isinstance(edge_cases, dict):
            converted = []
            for key, value in edge_cases.items():
                if isinstance(value, dict):
                    converted.append({'reference': key, **value})
                else:
                    converted.append({'reference': key, 'details': value})
            specification['edge_cases'] = converted
        elif not isinstance(edge_cases, list):
            specification['edge_cases'] = [{'details': edge_cases}]
        
        normalized_stories = []
        function_name = None
        signature_block = specification.get('function_signature') or specification.get('signature') or {}
        if isinstance(signature_block, dict):
            function_name = signature_block.get('name')
        if not function_name:
            function_name = specification.get('function_name') or specification.get('name', 'the function')
        function_label = f"{function_name}" if function_name else "the function"
        for index, story in enumerate(specification.get('user_stories') or [], start=1):
            if not isinstance(story, dict):
                continue
            story.setdefault('id', f"US-{index:02d}")
            story.setdefault('priority', 'P2')
            acceptance_entries = []
            has_substance = False
            for acceptance in story.get('acceptance') or []:
                if isinstance(acceptance, dict):
                    if acceptance.get('given') or acceptance.get('when') or acceptance.get('then'):
                        has_substance = True
                    acceptance_entries.append({
                        'given': acceptance.get('given', ''),
                        'when': acceptance.get('when', ''),
                        'then': acceptance.get('then', '')
                    })
            if not has_substance:
                acceptance_entries = []
            story['acceptance'] = acceptance_entries
            normalized_stories.append(story)
        specification['user_stories'] = normalized_stories
        
        normalized_success = []
        for index, criterion in enumerate(specification.get('success_criteria') or [], start=1):
            if not isinstance(criterion, dict):
                continue
            criterion.setdefault('id', f"SC-{index:02d}")
            criterion.setdefault('metric', '')
            criterion.setdefault('target', '')
            normalized_success.append(criterion)
        specification['success_criteria'] = normalized_success
        
        normalized_tests = []
        story_test_map: Dict[str, List[Dict[str, Any]]] = {}
        for test in specification.get('test_matrix') or []:
            if not isinstance(test, dict):
                continue
            normalized_entry = {
                'name': test.get('name') or test.get('scenario') or 'unnamed_scenario',
                'story_refs': test.get('story_refs') or test.get('stories') or [],
                'inputs': test.get('inputs') or {},
                'expected_output': test.get('expected_output'),
                'expected_exception': test.get('expected_exception'),
                'state_assertions': test.get('state_assertions') or []
            }
            normalized_tests.append(normalized_entry)
            for ref in normalized_entry['story_refs'] or []:
                story_test_map.setdefault(ref, []).append(normalized_entry)
        specification['test_matrix'] = normalized_tests
        
        story_success_map: Dict[str, List[Dict[str, Any]]] = {}
        for criterion in normalized_success:
            ref = (
                criterion.get('story_ref')
                or criterion.get('story_id')
                or criterion.get('story')
                or criterion.get('storyRefs')
            )
            if ref:
                story_success_map.setdefault(ref, []).append(criterion)
        
        def summarize_inputs(inputs: Any) -> str:
            if inputs is None:
                return "default inputs"
            if isinstance(inputs, dict):
                if not inputs:
                    return "default inputs"
                parts = []
                for key, value in list(inputs.items())[:4]:
                    try:
                        rendered = json.dumps(value)
                    except TypeError:
                        rendered = repr(value)
                    parts.append(f"{key}={rendered}")
                remaining = len(inputs) - len(parts)
                if remaining > 0:
                    parts.append(f"...(+{remaining} more)")
                return ", ".join(parts)
            if isinstance(inputs, list):
                preview = inputs[:4]
                try:
                    rendered = json.dumps(preview)
                except TypeError:
                    rendered = repr(preview)
                suffix = ""
                if len(inputs) > len(preview):
                    suffix = f" ...(+{len(inputs) - len(preview)} more)"
                return f"args={rendered}{suffix}"
            try:
                return json.dumps(inputs)
            except TypeError:
                return repr(inputs)
        
        def build_acceptance_from_test(test: Dict[str, Any]) -> Dict[str, str]:
            expected_exception = test.get('expected_exception')
            if expected_exception:
                then_text = f"Then it raises {expected_exception}"
            else:
                then_text = f"Then it returns {json.dumps(test.get('expected_output'))}"
            return {
                'given': f"Given {summarize_inputs(test.get('inputs', {}))}",
                'when': f"When {function_label} executes",
                'then': then_text
            }
        
        for story in normalized_stories:
            if story['acceptance']:
                continue
            story_id = story.get('id')
            acceptance_entries: List[Dict[str, str]] = []
            
            for test_entry in story_test_map.get(story_id, [])[:2]:
                acceptance_entries.append(build_acceptance_from_test(test_entry))
            
            if not acceptance_entries:
                for criterion in story_success_map.get(story_id, [])[:2]:
                    metric = criterion.get('metric') or "the documented metric"
                    target = criterion.get('target') or "the required target"
                    acceptance_entries.append({
                        'given': "Given compliant inputs",
                        'when': f"When {function_label} executes",
                        'then': f"Then it satisfies metric \"{metric}\" (target: {target})"
                    })
            
            if not acceptance_entries:
                fallback_narrative = story.get('narrative') or story.get('title') or "the user scenario"
                acceptance_entries.append({
                    'given': f"Given {fallback_narrative}",
                    'when': f"When {function_label} executes",
                    'then': "Then the story outcome is achieved"
                })
            
            story['acceptance'] = acceptance_entries
        
        return specification
    
    def _detect_spec_drift(
        self,
        specification: Dict[str, Any],
        func_info: Dict[str, Any],
        code_analysis: Dict[str, Any]
    ) -> List[str]:
        """Identify key mismatches between generated spec and original implementation."""
        issues: List[str] = []
        source_code = func_info.get('source_code', '')
        if not source_code:
            return issues
        
        try:
            tree = ast.parse(textwrap.dedent(source_code))
        except SyntaxError:
            return issues
        
        # Check for stateful mutations
        stateful_attrs = self._detect_stateful_mutations(tree)
        spec_side_effects = specification.get('side_effects')
        spec_claims_no_side_effects = False
        if isinstance(spec_side_effects, dict):
            values = " ".join(str(v) for v in spec_side_effects.values() if isinstance(v, str)).lower()
            if values:
                spec_claims_no_side_effects = "no side effect" in values or values.strip() in {"none", "no side effects"}
        elif isinstance(spec_side_effects, str):
            lowered = spec_side_effects.lower()
            spec_claims_no_side_effects = "no side effect" in lowered or lowered.strip() in {"none", "no side effects"}
        
        if stateful_attrs and spec_claims_no_side_effects:
            formatted = ", ".join(sorted(stateful_attrs))
            issues.append(f"Spec claims no side effects but code mutates instance attributes: {formatted}.")
        
        # Check return type consistency
        return_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Return)]
        has_returns = len(return_nodes) > 0
        spec_return_type = specification.get('return_type', '')
        if has_returns and not spec_return_type:
            issues.append("Code has return statements but spec doesn't specify return_type.")
        
        # Check for exception handling
        raise_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Raise)]
        has_exceptions = len(raise_nodes) > 0
        spec_error_handling = specification.get('error_handling', '')
        if has_exceptions and (not spec_error_handling or 'raise' not in str(spec_error_handling).lower()):
            issues.append("Code raises exceptions but spec doesn't document error handling properly.")
        
        # Check for loop structures
        loop_nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While, ast.AsyncFor))]
        has_loops = len(loop_nodes) > 0
        spec_internal_logic = specification.get('internal_logic', '')
        if has_loops and spec_internal_logic:
            logic_str = str(spec_internal_logic).lower()
            if 'loop' not in logic_str and 'iterate' not in logic_str and 'for' not in logic_str:
                issues.append("Code contains loops but spec internal_logic doesn't mention iteration.")
        
        # Check parameter count
        func_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_node = node
                break
        
        if func_node:
            actual_params = len(func_node.args.args)
            spec_params = specification.get('parameters', [])
            if isinstance(spec_params, list):
                spec_param_count = len([p for p in spec_params if isinstance(p, dict) and p.get('name') != 'self'])
            else:
                spec_param_count = 0
            
            if actual_params != spec_param_count and 'self' not in [p.get('name', '') for p in (spec_params if isinstance(spec_params, list) else [])]:
                issues.append(f"Parameter count mismatch: code has {actual_params} parameters, spec has {spec_param_count}.")
        
        # Detect missing state mention in english summary for stateful methods
        english_summary = specification.get('english_summary', '')
        if stateful_attrs and english_summary:
            lowered_summary = english_summary.lower()
            mentions_state = any(attr.lower() in lowered_summary for attr in stateful_attrs)
            if not mentions_state:
                issues.append("English summary does not mention the instance state changes present in the code.")
        
        # Detect discrepancy between documented return type and actual return expressions
        actual_return_kinds = self._infer_return_kinds(tree)
        return_desc = specification.get('return_type') or specification.get('return_value')
        if actual_return_kinds and return_desc:
            desc_text = ""
            if isinstance(return_desc, dict):
                desc_text = " ".join(str(v) for v in return_desc.values() if isinstance(v, str))
            elif isinstance(return_desc, str):
                desc_text = return_desc
            desc_text = desc_text.lower()
            if "dict" in desc_text or "dictionary" in desc_text:
                if "dict" not in actual_return_kinds:
                    issues.append("Spec describes a dictionary return, but the implementation returns non-dictionary values.")
            elif "tuple" in desc_text:
                if "tuple" not in actual_return_kinds:
                    issues.append("Spec describes a tuple return, but the implementation returns different structures.")
            elif "list" in desc_text:
                if "list" not in actual_return_kinds:
                    issues.append("Spec describes a list return, but the implementation returns different structures.")
        
        return issues
    
    def _detect_stateful_mutations(self, tree: ast.AST) -> Set[str]:
        """Capture instance attributes mutated within the function body."""
        mutated: Set[str] = set()
        
        class MutationVisitor(ast.NodeVisitor):
            def visit_Assign(self, node: ast.Assign):
                for target in node.targets:
                    attr = self._extract_self_attr(target)
                    if attr:
                        mutated.add(attr)
                self.generic_visit(node)
            
            def visit_AugAssign(self, node: ast.AugAssign):
                attr = self._extract_self_attr(node.target)
                if attr:
                    mutated.add(attr)
                self.generic_visit(node)
            
            def visit_Call(self, node: ast.Call):
                attr = self._extract_mutating_call(node.func)
                if attr:
                    mutated.add(attr)
                self.generic_visit(node)
            
            def _extract_self_attr(self, node: ast.AST) -> Optional[str]:
                if isinstance(node, ast.Attribute):
                    base = node.value
                    if isinstance(base, ast.Name) and base.id == 'self':
                        return node.attr
                    if isinstance(base, ast.Attribute):
                        # Handle nested attributes like self.state.current
                        while isinstance(base, ast.Attribute):
                            if isinstance(base.value, ast.Name) and base.value.id == 'self':
                                return base.attr
                            base = base.value
                return None
            
            def _extract_mutating_call(self, node: ast.AST) -> Optional[str]:
                if not isinstance(node, ast.Attribute):
                    return None
                method_name = node.attr
                mutating_methods = {
                    'append', 'extend', 'insert', 'remove', 'pop', 'clear', 'add',
                    'update', 'setdefault', 'discard', 'push', 'put', 'appendleft',
                    'popleft'
                }
                if isinstance(node.value, ast.Attribute):
                    base_attr = self._extract_self_attr(node.value)
                    if base_attr and method_name in mutating_methods:
                        return base_attr
                elif isinstance(node.value, ast.Name) and node.value.id == 'self' and method_name in mutating_methods:
                    return method_name
                return None
        
        MutationVisitor().visit(tree)
        return mutated
    
    def _infer_return_kinds(self, tree: ast.AST) -> Set[str]:
        """Infer coarse-grained return structures from the function."""
        kinds: Set[str] = set()
        
        class ReturnVisitor(ast.NodeVisitor):
            def visit_Return(self, node: ast.Return):
                value = node.value
                if isinstance(value, ast.Dict):
                    kinds.add('dict')
                elif isinstance(value, ast.List):
                    kinds.add('list')
                elif isinstance(value, ast.Tuple):
                    kinds.add('tuple')
                elif isinstance(value, ast.Call):
                    func = value.func
                    if isinstance(func, ast.Name):
                        if func.id in {'dict', 'list', 'tuple'}:
                            kinds.add(func.id)
                self.generic_visit(node)
        
        ReturnVisitor().visit(tree)
        return kinds


class CodeRegenerationNode(BaseNode):
    """Regenerates code from specifications"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from utils.call_llm import call_llm
        self.call_llm = call_llm
        self.few_shot_enhancer = FewShotPromptEnhancer()
    
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
                # Get original code from context (stored separately, NOT in specification)
                original_code = context.get('original_code', {}).get(func_id, '')
                if not original_code:
                    # Fallback: try to get from func_info if available
                    func_info = context.get('function_info', {}).get(func_id, {})
                    original_code = func_info.get('source_code', '')
                
                code = self._regenerate_code(spec_data['specification'], original_code=original_code)
                
                if code:
                    regenerated_code[func_id] = {
                        'code': code,
                        'function_name': spec_data['function_name'],
                        'file_path': spec_data['file_path']
                    }
                    print(f"        ✓ Code regenerated ({len(code)} chars)")
                else:
                    print(f"        ✗ ERROR: Failed to regenerate code - returned None/empty")
            
            except Exception as e:
                error_msg = str(e)
                if self.config.get('verbose'):
                    traceback.print_exc()
                print(f"        ✗ ERROR: Exception during regeneration: {error_msg[:100]}")
                continue
        
        context['regenerated_code'] = regenerated_code
        print(f"    Regenerated {len(regenerated_code)} functions")
        
        return context
    
    def _regenerate_code(self, specification: Dict[str, Any], max_retries: int = 2, original_code: str = None) -> Optional[str]:
        """Regenerate code from specification with retry logic and example-driven enhancement"""
        
        # Hybrid last-resort: use exact code directly when spec says so (guarantees 100% similarity)
        if specification.get('hybrid_use_exact_code'):
            additions = specification.get('hybrid_code_additions', [])
            if additions and isinstance(additions[-1], str):
                return additions[-1].strip()
        
        # Check if this is a class method
        is_class_method = 'class_context' in specification
        
        # DO NOT store original code in specification - it must never be included
        # Original code is stored separately in context for similarity comparison only
        
        # Add example-driven prompt enhancement
        from agents.example_driven import ExampleDrivenSpecEnhancer
        from agents.constraint_propagation import ConstraintPropagator
        example_enhancer = ExampleDrivenSpecEnhancer()
        constraint_propagator = ConstraintPropagator()
        example_section = example_enhancer.generate_example_based_prompt_enhancement(specification)

        bctx_guide = ""
        if specification.get("bounded_context_bundle"):
            bctx_guide = (
                "\n\nBOUNDED CONTEXT (read-only):\n"
                "The Specification JSON includes `bounded_context_bundle` excerpts from sibling callees and scoped definitions. "
                "Use them to infer behaviour — do NOT paste that text verbatim; synthesize equivalent logic in your answer.\n"
            )

        # Extract and propagate constraints (original_code passed separately, not in spec)
        constraints = constraint_propagator.extract_constraints(specification, original_code or '')
        constraint_section = constraint_propagator.generate_constraint_prompt_section(constraints) if constraints else ""
        
        # Check if we have path analysis for divide-and-conquer regeneration
        path_analysis = specification.get('path_analysis', {})
        has_paths = path_analysis.get('paths', [])
        
        path_guidance = ""
        if has_paths:
            path_guidance = f"\n\nDIVIDE-AND-CONQUER REGENERATION GUIDANCE:\n"
            path_guidance += f"This function has {len(has_paths)} distinct execution paths identified:\n"
            for i, path in enumerate(has_paths[:5], 1):
                path_type = path.get('type', 'unknown')
                conditions = path.get('conditions', [])
                path_guidance += f"  Path {i} ({path_type}): {', '.join(conditions[:2]) if conditions else 'default path'}\n"
            path_guidance += "\nEnsure your regenerated code handles ALL these paths correctly. "
            path_guidance += "The control flow structure must match the path analysis.\n"
        
        # Extract variable names from specification for emphasis
        variable_names_section = ""
        var_names = specification.get('variable_names', [])
        # Also check 'variables' field for backward compatibility
        if not var_names:
            var_names = specification.get('variables', [])
        
        if var_names:
            var_list = []
            exact_names = []
            for var in var_names[:20]:  # Increased limit
                if isinstance(var, dict):
                    name = var.get('name', '')
                    purpose = var.get('purpose', '')
                    preserve = var.get('preserve_exact_name', False)
                    if name:
                        if preserve:
                            exact_names.append(name)
                        var_list.append(f"{name} ({purpose})" if purpose else name)
                elif isinstance(var, str):
                    var_list.append(var)
                    exact_names.append(var)
            
            if var_list:
                variable_names_section = f"\n\nCRITICAL: Use these EXACT variable names (DO NOT RENAME):\n"
                variable_names_section += "\n".join(f"  - {v}" for v in var_list[:15])
                if exact_names:
                    variable_names_section += f"\n\nMOST CRITICAL VARIABLES (must match exactly):\n"
                    variable_names_section += "\n".join(f"  - {name}" for name in exact_names[:10])
                variable_names_section += "\n\nDO NOT use synonyms, abbreviations, or variations of these names.\n"
                variable_names_section += "Use the EXACT names as specified above, preserving case and spelling.\n"
        
        # Extract control flow details
        control_flow_section = ""
        control_flow = specification.get('control_flow', '')
        if control_flow:
            control_flow_section = f"\n\nCRITICAL: Control flow structure (MUST MATCH EXACTLY):\n{control_flow}\n"
            control_flow_section += "Match this EXACT control flow structure in your regenerated code:\n"
            control_flow_section += "- Same nesting levels (same indentation depth)\n"
            control_flow_section += "- Same if/else/elif structure\n"
            control_flow_section += "- Same loop types (for vs while)\n"
            control_flow_section += "- Same branch conditions (if conditions must match)\n"
            control_flow_section += "- Same return patterns (early returns, final returns)\n"
        
        # Extract detailed control flow if available
        detailed_control_flow = specification.get('detailed_control_flow', [])
        if detailed_control_flow and not control_flow:
            control_flow_section = "\n\nCRITICAL: Detailed Control Flow Structure (MUST MATCH):\n"
            for item in detailed_control_flow[:10]:
                indent = "  " * item.get('nesting_level', 0)
                flow_type = item.get('type', '').upper()
                condition = item.get('condition', '')
                control_flow_section += f"{indent}{flow_type}"
                if condition:
                    control_flow_section += f" - {condition}"
                control_flow_section += "\n"
            control_flow_section += "\nMatch this EXACT nesting structure and control flow in your code.\n"
        
        if is_class_method:
            class_ctx = specification['class_context']
            method_name = specification.get('function_name', 'method')
            class_attrs = class_ctx.get('class_attributes', [])
            class_name = class_ctx.get('class_name', 'Unknown')
            
            # DO NOT include original code - regenerate purely from specification
            original_method_ref = ""
            
            # Add causal analysis guidance (NOVEL)
            causal_analysis_section = ""
            if 'causal_insights' in specification and specification.get('causal_insights'):
                causal_analysis_section = f"\n\n{specification['causal_insights']}\n"
            elif 'causal_analysis' in specification:
                causal_spec = specification.get('causal_analysis', {})
                minimal_elements = causal_spec.get('minimal_elements', [])
                if minimal_elements:
                    causal_analysis_section = f"\n\nCAUSAL ANALYSIS:\n"
                    causal_analysis_section += f"The following {len(minimal_elements)} elements are NECESSARY for correct behavior.\n"
                    causal_analysis_section += f"Minimal element IDs: {', '.join(minimal_elements[:10])}\n"
                    causal_analysis_section += "Ensure these elements are properly represented in regenerated code.\n"
            
            # Initialize slicing and logical deletion sections
            slicing_guidance_section = ""
            slicing_data = specification.get('slicing_analysis', {})
            if slicing_data and slicing_data.get('slices'):
                slices = slicing_data['slices']
                slicing_guidance_section = "\n\nSLICE-BY-SLICE REQUIREMENTS (Derived from Program Slicing):\n"
                for slice_info in slices[:6]:
                    criterion = slice_info.get('criterion', {})
                    desc = criterion.get('description') or criterion.get('target_code') or criterion.get('type')
                    line_range = slice_info.get('line_range', (0, 0))
                    guards = slice_info.get('guard_conditions', [])
                    slicing_guidance_section += (
                        f"- {slice_info.get('slice_id')}: {desc} "
                        f"(lines {line_range[0]}-{line_range[1]}); "
                        f"Guards: {', '.join(guards[:2]) if guards else 'default path'}\n"
                    )
                slicing_guidance_section += "Each user story/test must map to one of these slices.\n"
            
            logical_deletion_section = ""
            logical_plan = specification.get('logical_deletion')
            if logical_plan:
                logical_deletion_section = "\n\nLOGICAL DELETION GUARDRAILS:\n"
                logical_deletion_section += f"- Preserve critical lines: {logical_plan.get('critical_lines', [])[:12]}\n"
                logical_deletion_section += f"- Remove/avoid resurrecting lines: {logical_plan.get('deletable_lines', [])[:12]}\n"
                if logical_plan.get('slice_assertions'):
                    logical_deletion_section += "Required slice assertions:\n"
                    for assertion in logical_plan['slice_assertions'][:5]:
                        logical_deletion_section += f"  - {assertion.get('assertion_id', '')}: When {assertion.get('precondition', '')} expect {assertion.get('expected_effect', '')}\n"
            
            # Specification JSON for regen respects ``regeneration_spec_json_char_budget`` and richer keys.
            spec_json = build_regeneration_spec_json_blob(specification, self.config)

            hybrid_guidance_section = ""
            if specification.get('hybrid_code_additions') and not specification.get('hybrid_use_exact_code'):
                hybrid_guidance_section = (
                    "\n\nMINIMAL REFERENCE SNIPPETS (hybrid_code_additions):\n"
                    "These lines are constraints from the reference implementation, not a full solution.\n"
                    "Synthesize a complete function body that satisfies EVERY snippet (same branches, messages, calls).\n"
                    "Do not paste them as dead code—integrate equivalent logic so AST structure matches.\n"
                )
            if specification.get('hybrid_diff_summary'):
                hybrid_guidance_section += "\n\nCRITICAL DIFFERENCES TO FIX:\n" + "\n".join(
                    f"  - {d}" for d in specification['hybrid_diff_summary'][:12]
                )
            gaps_cm = specification.get('structural_implementation_gaps') or []
            if gaps_cm:
                hybrid_guidance_section += "\n\nSTRUCTURAL IMPLEMENTATION GAPS (failure-driven — satisfy all):\n" + "\n".join(
                    f"  - {g}" for g in gaps_cm[:15]
                )
            
            prompt = f"""
Generate ONLY the method code (not the entire class) based on this specification.

Specification:
    {spec_json}
{bctx_guide}
{original_method_ref}
{path_guidance}{variable_names_section}{control_flow_section}{example_section}{constraint_section}{causal_analysis_section}{slicing_guidance_section}{logical_deletion_section}{hybrid_guidance_section}
CRITICAL REQUIREMENTS FOR CLASS METHOD (MUST FOLLOW EXACTLY):
    - Generate ONLY the method definition: def {method_name}(self, ...):
        - Do NOT include class definition, imports, or any other code
- Include 'self' as first parameter
- The method belongs to class: {class_name}
- Available class attributes: {', '.join(class_attrs) if class_attrs else 'none documented'}
- Use EXACT variable names as specified in 'variable_names' section (do NOT rename, do NOT use synonyms)
- Match the exact control flow structure from the original code (same if/else/loop structure)
- Preserve method logic patterns from original (same algorithmic approach)
- Handle all example inputs/outputs correctly (same behavior)
- Access instance attributes via self.attribute_name (exact attribute names)
- If method modifies state, use self.attribute = value (same state modifications)
- Variable name matching is CRITICAL - use exact names from specification, NOT variations

PYTHON SYNTAX REQUIREMENTS (CRITICAL - MUST BE VALID PYTHON):
    - Use 4 spaces for indentation (NO tabs, NO mixed indentation)
- All control structures (if, for, while, def, class) MUST end with a colon ':'
- Properly close all brackets, parentheses, and braces
- No trailing whitespace or extra blank lines at the end
- Function body must be indented exactly 4 spaces from 'def' line
- Nested blocks must increase indentation by exactly 4 spaces per level
- All statements must be properly indented relative to their parent block

CODE STRUCTURE REQUIREMENTS:
    - Start with: def {method_name}(self, ...):
        - Proper indentation (4 spaces for method body)
        - If original method had leading spaces before 'def', preserve them EXACTLY
        - Docstring must be indented exactly 4 spaces from 'def' line (same as method body)
- Match original method's return pattern exactly
- Include docstring if original had one, with EXACT indentation matching original

OUTPUT FORMAT:
    - Return ONLY valid Python code
- NO markdown code blocks (no ```python or ```)
- NO explanatory text before or after the code
- NO comments explaining what the code does
- Code must be syntactically valid and parseable by Python's AST parser

Generate ONLY the method code, nothing else. No explanations, no markdown formatting, just the Python method.
"""
        else:
            # Add causal analysis guidance (NOVEL)
            causal_analysis_section = ""
            if 'causal_insights' in specification and specification.get('causal_insights'):
                causal_analysis_section = f"\n\n{specification['causal_insights']}\n"
            elif 'causal_analysis' in specification:
                causal_spec = specification.get('causal_analysis', {})
                minimal_elements = causal_spec.get('minimal_elements', [])
                if minimal_elements:
                    causal_analysis_section = f"\n\nCAUSAL ANALYSIS:\n"
                    causal_analysis_section += f"The following {len(minimal_elements)} elements are NECESSARY for correct behavior.\n"
                    causal_analysis_section += f"Minimal element IDs: {', '.join(minimal_elements[:10])}\n"
                    causal_analysis_section += "Ensure these elements are properly represented in regenerated code.\n"
            
            # Add enhanced documentation sections
            enhanced_description_section = ""
            if 'detailed_english_description' in specification:
                enhanced_description_section = f"\n\nDETAILED ENGLISH DESCRIPTION:\n{specification['detailed_english_description']}\n"
            
            step_by_step_section = ""
            if 'detailed_step_by_step_logic' in specification:
                steps = specification['detailed_step_by_step_logic']
                if steps:
                    step_by_step_section = "\n\nSTEP-BY-STEP LOGIC FLOW:\n"
                    for step in steps[:20]:  # Limit to first 20 steps
                        indent = "  " * step.get('indent_level', 0)
                        step_by_step_section += f"{indent}Step {step.get('step')} ({step.get('type')}): {step.get('description')}\n"
            
            detailed_control_flow_section = ""
            if 'detailed_control_flow' in specification:
                flow = specification['detailed_control_flow']
                if flow:
                    detailed_control_flow_section = "\n\nDETAILED CONTROL FLOW STRUCTURE:\n"
                    for item in flow[:15]:  # Limit to first 15 items
                        indent = "  " * item.get('nesting_level', 0)
                        flow_desc = f"{item.get('type').upper()}"
                        if item.get('condition'):
                            flow_desc += f" - {item.get('condition')}"
                        detailed_control_flow_section += f"{indent}{flow_desc}\n"
            
            detailed_variable_section = ""
            if 'detailed_variable_usage' in specification:
                var_usage = specification['detailed_variable_usage']
                if var_usage:
                    detailed_variable_section = "\n\nDETAILED VARIABLE USAGE:\n"
                    for var_name, usages in list(var_usage.items())[:10]:  # Limit to 10 variables
                        detailed_variable_section += f"  {var_name}:\n"
                        for usage in usages[:3]:  # First 3 usages per variable
                            detailed_variable_section += f"    Line {usage.get('line')}: {usage.get('context')} ({usage.get('usage_type')})\n"
        
        # Note: slicing_guidance_section and logical_deletion_section are initialized earlier for class methods
        # For non-class methods, initialize them here if not already set
        if not is_class_method:
            slicing_guidance_section = ""
            slicing_data = specification.get('slicing_analysis', {})
            if slicing_data and slicing_data.get('slices'):
                slices = slicing_data['slices']
                slicing_guidance_section = "\n\nSLICE-BY-SLICE REQUIREMENTS (Derived from Program Slicing):\n"
                for slice_info in slices[:6]:
                    criterion = slice_info.get('criterion', {})
                    desc = criterion.get('description') or criterion.get('target_code') or criterion.get('type')
                    line_range = slice_info.get('line_range', (0, 0))
                    guards = slice_info.get('guard_conditions', [])
                    slicing_guidance_section += (
                        f"- {slice_info.get('slice_id')}: {desc} "
                        f"(lines {line_range[0]}-{line_range[1]}); "
                        f"Guards: {', '.join(guards[:2]) if guards else 'default path'}\n"
                    )
                slicing_guidance_section += "Each user story/test must map to one of these slices.\n"
            
            logical_deletion_section = ""
            logical_plan = specification.get('logical_deletion')
            if logical_plan:
                logical_deletion_section = "\n\nLOGICAL DELETION GUARDRAILS:\n"
                logical_deletion_section += f"- Preserve critical lines: {logical_plan.get('critical_lines', [])[:12]}\n"
                logical_deletion_section += f"- Remove/avoid resurrecting lines: {logical_plan.get('deletable_lines', [])[:12]}\n"
                if logical_plan.get('slice_assertions'):
                    logical_deletion_section += "Required slice assertions:\n"
                    for assertion in logical_plan['slice_assertions'][:5]:
                        logical_deletion_section += (
                            f"  * {assertion['assertion_id']}: If {assertion['precondition']} "
                            f"then {assertion['expected_effect']} (lines {assertion['line_range'][0]}-{assertion['line_range'][1]}).\n"
                        )
                logical_deletion_section += logical_plan.get('summary', '')
            
            # Specification JSON for regen respects ``regeneration_spec_json_char_budget`` and richer keys.
            spec_json = build_regeneration_spec_json_blob(specification, self.config)

            hybrid_guidance_section = ""
            if specification.get('hybrid_code_additions') and not specification.get('hybrid_use_exact_code'):
                hybrid_guidance_section = (
                    "\n\nMINIMAL REFERENCE SNIPPETS (hybrid_code_additions):\n"
                    "These lines are constraints from the reference implementation, not a full solution.\n"
                    "Synthesize a complete function body that satisfies EVERY snippet (same branches, messages, calls).\n"
                    "Do not paste them as dead code—integrate equivalent logic so AST structure matches.\n"
                )
            if specification.get('hybrid_diff_summary'):
                hybrid_guidance_section += "\n\nCRITICAL DIFFERENCES TO FIX (regenerated must address):\n"
                for d in specification['hybrid_diff_summary'][:12]:
                    hybrid_guidance_section += f"  - {d}\n"
            gaps_nc = specification.get('structural_implementation_gaps') or []
            if gaps_nc:
                hybrid_guidance_section += "\n\nSTRUCTURAL IMPLEMENTATION GAPS (failure-driven — satisfy all):\n" + "\n".join(
                    f"  - {g}" for g in gaps_nc[:15]
                )
            
            # Add docstring section if specification has one (for non-class methods)
            docstring_section = ""
            if specification.get('docstring'):
                docstring_text = specification['docstring']
                docstring_section = f"\n\nCRITICAL DOCSTRING REQUIREMENT:\n"
                docstring_section += f"The original code has this EXACT docstring - you MUST include it immediately after the function definition:\n"
                docstring_section += f'    \"\"\"{docstring_text}\"\"\"\n'
                docstring_section += "Do NOT modify, shorten, or paraphrase this docstring. Include it EXACTLY as shown above.\n"
            
            prompt = f"""
Generate Python code based on the following detailed specification. The code should match the specification EXACTLY.

Specification:
    {spec_json}
{bctx_guide}
{enhanced_description_section}{step_by_step_section}{detailed_control_flow_section}{detailed_variable_section}{path_guidance}{variable_names_section}{control_flow_section}{example_section}{constraint_section}{causal_analysis_section}
{slicing_guidance_section}{logical_deletion_section}{hybrid_guidance_section}{docstring_section}
CRITICAL REQUIREMENTS (MUST FOLLOW EXACTLY - THESE DETERMINE SIMILARITY SCORES):

VARIABLE NAMES (MOST CRITICAL):
    - Use EXACT variable names as specified in 'variable_names' and 'detailed_variable_usage' sections
- DO NOT rename variables - if spec says 'result', use 'result' NOT 'output', 'res', 'value', 'ret', etc.
- DO NOT use synonyms - if spec says 'data', use 'data' NOT 'items', 'list', 'arr', 'collection'
- DO NOT abbreviate - if spec says 'index', use 'index' NOT 'idx', 'i', 'ind'
- DO NOT change loop variables - if spec shows 'i', use 'i' NOT 'idx', 'j', 'k', 'item'
- Variable name matching is THE MOST IMPORTANT factor for similarity scores
- Every variable name must match exactly - case-sensitive, spelling-sensitive

CONTROL FLOW (CRITICAL):
    - Follow the step-by-step logic flow EXACTLY as documented in 'detailed_step_by_step_logic'
- Match the EXACT control flow structure from 'detailed_control_flow' (same nesting levels, same if/else/loop structure)
- CRITICAL: Preserve exact nesting depth - check 'max_nesting_depth' in specification
{f"- Maximum nesting depth: {specification.get('max_nesting_depth', 'not specified')} levels - preserve this EXACTLY" if specification.get('max_nesting_depth') else ""}
- Same nesting depth - if original has N levels of indentation, regenerated must have N levels (each level = 4 spaces)
- Same branch structure - if original has 'if-else', regenerated must have 'if-else' (not 'if-elif-else')
- Same loop types - if original uses 'for', regenerated must use 'for' (not 'while')
- Same return patterns - if original has early returns, regenerated must have same early returns
- Indentation is CRITICAL: Level 0 = 0 spaces, Level 1 = 4 spaces, Level 2 = 8 spaces, Level 3 = 12 spaces, etc.

CODE STRUCTURE:
    - Preserve all comments and docstrings exactly as specified
- CRITICAL: If the original code had a docstring, the regenerated code MUST include the EXACT same docstring (word-for-word, including formatting)
- Preserve inline comments exactly as they appear in the original code
- Use the exact function signature with exact parameter names (no renaming)
- Match return types and values exactly (same return patterns)
- Handle all example inputs/outputs correctly (same behavior)
- The detailed English description provides the high-level overview - ensure your code matches it
- Docstring formatting matters: preserve triple quotes style (\"\"\" or ''') and indentation

PYTHON SYNTAX REQUIREMENTS (CRITICAL - MUST BE VALID PYTHON):
    - Use 4 spaces for indentation (NO tabs, NO mixed indentation)
- All control structures (if, for, while, def, class, try, except) MUST end with a colon ':'
- Properly close all brackets, parentheses, and braces - every '(' needs ')', every '[' needs ']', every '{{' needs '}}'
- No trailing whitespace or extra blank lines at the end
- Function body must be indented exactly 4 spaces from 'def' line
- Nested blocks must increase indentation by exactly 4 spaces per level
- All statements must be properly indented relative to their parent block
- No syntax errors - code must be parseable by Python's AST parser
- String quotes must be properly closed (matching single or double quotes)

OUTPUT FORMAT:
    - Return ONLY valid Python code
- NO markdown code blocks (no ```python or ```)
- NO explanatory text before or after the code
- NO comments explaining what the code does
- Code must be syntactically valid and parseable by Python's AST parser
- Start directly with 'def function_name(...):' - no preamble

Generate only the Python function code, no explanations or markdown formatting.
The code should be complete, runnable, and syntactically valid.
"""
        
        # Enhance prompt with few-shot examples and variable name enforcement
        # Only add few-shot examples if prompt isn't too long (avoid breaking complex prompts)
        try:
            if len(prompt) < 20000:  # Only add examples if prompt isn't already very long
                prompt = self.few_shot_enhancer.add_examples_to_prompt(prompt, specification)
                prompt = self.few_shot_enhancer.add_variable_name_enforcement(prompt, specification)
        except Exception as e:
            # If few-shot enhancement fails, continue without it (don't break regeneration)
            if self.config.get('verbose'):
                print(f"        WARNING: Few-shot enhancement failed: {e}")
        
        # Retry logic for code regeneration
        last_error = None
        for attempt in range(max_retries):
            try:
                # Check prompt length - if too long, use simplified prompt immediately
                if len(prompt) > 25000:  # Very long prompts: simplify to stay within context and latency budget
                    print(f"        WARNING: Prompt too long ({len(prompt)} chars), using simplified prompt...")
                    if attempt == 0:
                        # First attempt with long prompt failed, use simplified immediately
                        prompt = self._create_simplified_prompt(specification, is_class_method)
                        import time
                        time.sleep(2)
                    else:
                        # Already using simplified, try ultra-simple
                        prompt = self._create_ultra_simple_prompt(specification, is_class_method)
                        import time
                        time.sleep(2)
                elif len(prompt) > 20000:
                    # Truncate verbose sections but keep prompt
                    print(f"        WARNING: Prompt long ({len(prompt)} chars), truncating verbose sections...")
                    spec_copy = specification.copy()
                    if 'user_stories' in spec_copy and len(str(spec_copy['user_stories'])) > 5000:
                        spec_copy['user_stories'] = spec_copy['user_stories'][:3]  # Keep first 3
                    if 'test_matrix' in spec_copy and len(str(spec_copy['test_matrix'])) > 3000:
                        spec_copy['test_matrix'] = spec_copy['test_matrix'][:5]  # Keep first 5
                    if 'detailed_step_by_step_logic' in spec_copy and len(str(spec_copy['detailed_step_by_step_logic'])) > 2000:
                        spec_copy['detailed_step_by_step_logic'] = str(spec_copy['detailed_step_by_step_logic'])[:2000]
                    prompt = prompt.replace(json.dumps(specification, indent=2), json.dumps(spec_copy, indent=2))
                
                response = self.call_llm(prompt, system=SYSTEM_CODE_REGENERATION)
                
                # Check if response is empty or None
                if not response or not response.strip():
                    print(f"        WARNING: Empty response from LLM (attempt {attempt + 1}/{max_retries}, prompt_len={len(prompt)})")
                    if attempt < max_retries - 1:
                        # Try simpler prompt on retry
                        prompt = self._create_simplified_prompt(specification, is_class_method)
                        print(f"        Trying simplified prompt (len={len(prompt)})...")
                        # Add delay for rate limiting
                        import time
                        time.sleep(3)
                        continue
                    # Last attempt: try even simpler prompt
                    if attempt == max_retries - 1:
                        prompt = self._create_ultra_simple_prompt(specification, is_class_method)
                        print(f"        Trying ultra-simple prompt (len={len(prompt)})...")
                        import time
                        time.sleep(2)
                        response = self.call_llm(prompt, system=SYSTEM_CODE_REGENERATION)
                        if response and response.strip():
                            code = self._clean_generated_code(response)
                            if code and len(code.strip()) >= 10:
                                return code
                            else:
                                print(f"        WARNING: Ultra-simple prompt response cleaned to {len(code.strip()) if code else 0} chars")
                        else:
                            print(f"        WARNING: Ultra-simple prompt also returned empty")
                    return None
                
                code = self._clean_generated_code(response)
                
                # Validate that we got actual code
                if not code or len(code.strip()) < 10:
                    print(f"        WARNING: Cleaned code too short ({len(code.strip()) if code else 0} chars) (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        # Try simpler prompt on retry
                        prompt = self._create_simplified_prompt(specification, is_class_method)
                        import time
                        time.sleep(2)
                        continue
                    return None
                
                # If this is a class method and we got a full class, extract just the method
                if is_class_method and 'class ' in code.lower():
                    code = self._extract_method_from_class(code, specification.get('function_name', ''))
                
                # Post-regeneration validation: Check variable names match specification
                var_validation = self._validate_variable_names(code, specification)
                if not var_validation['valid'] and var_validation['missing_vars']:
                    missing = ', '.join(var_validation['missing_vars'][:5])
                    if attempt < max_retries - 1:
                        print(f"        WARNING: Missing critical variables: {missing}")
                        # Add variable name feedback to prompt
                        prompt += f"\n\nPREVIOUS ATTEMPT FAILED: Missing required variables: {missing}\nYou MUST use these EXACT variable names in your code."
                        continue
                
                # Validate and automatically fix code using CodeValidator
                from agents.code_validator import CodeValidator
                validator = CodeValidator()
                function_name = specification.get('function_name', '')
                validated_code, validation_issues, is_valid = validator.validate_and_fix(code, function_name)
                
                if validation_issues:
                    validation_summary = validator.get_validation_summary()
                    if self.config.get('verbose'):
                        print(f"        Validation: {validation_summary}")
                
                if is_valid:
                    return validated_code
                else:
                    # If validation failed, try to get more info for retry
                    error_issues = [i for i in validation_issues if i.severity == 'error']
                    if error_issues and attempt < max_retries - 1:
                        error_messages = [f"{i.message} (line {i.line})" if i.line else i.message for i in error_issues[:3]]
                        prompt += f"\n\nCRITICAL: Previous attempt had validation errors that must be fixed:\n" + "\n".join(f"- {msg}" for msg in error_messages)
                        prompt += "\n\nEnsure your code:\n- Has proper indentation (4 spaces, no tabs)\n- All control structures end with ':'\n- All brackets/parentheses are properly closed\n- Is valid Python syntax that can be parsed by AST"
                        last_error = "; ".join(error_messages)
                        import time
                        time.sleep(2)
                        continue
                    
                    # Last attempt: return validated code even if not perfect (might be fixable later)
                    if validated_code and len(validated_code.strip()) >= 10:
                        print(f"        WARNING: Returning code with validation issues (attempt {attempt + 1}/{max_retries})")
                        return validated_code
                    
                    return None
                    
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
                    continue
                return None
        
        return None
    
    def _validate_variable_names(self, code: str, specification: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that regenerated code uses required variable names from specification"""
        result = {
            'valid': True,
            'missing_vars': [],
            'extra_vars': []
        }
        
        try:
            # Extract required variable names from specification
            required_vars = set()
            var_names = specification.get('variable_names', [])
            for var in var_names[:20]:  # Check top 20 most critical
                if isinstance(var, dict):
                    name = var.get('name', '')
                    if var.get('preserve_exact_name', False) and name:
                        required_vars.add(name)
                elif isinstance(var, str) and var.strip():
                    required_vars.add(var.strip())
            
            # If no critical variables specified, skip validation
            if not required_vars:
                return result
            
            # Extract actual variable names from regenerated code
            tree = ast.parse(code)
            actual_vars = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    actual_vars.add(node.id)
                elif isinstance(node, ast.FunctionDef):
                    # Include parameters
                    for arg in node.args.args:
                        actual_vars.add(arg.arg)
            
            # Check for missing critical variables
            missing = required_vars - actual_vars
            if missing:
                result['valid'] = False
                result['missing_vars'] = list(missing)
            
        except Exception:
            # If validation fails, don't block regeneration
            pass
        
        return result
    
    def _extract_method_from_class(self, code: str, method_name: str) -> str:
        """Extract just the method from a full class definition, preserving indentation"""
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == method_name:
                            # Extract this method's source with original indentation
                            lines = code.splitlines()
                            start = item.lineno - 1
                            end = item.end_lineno if hasattr(item, 'end_lineno') else start + 20
                            method_lines = lines[start:end]
                            
                            # Preserve original indentation
                            return '\n'.join(method_lines)
            
            # If extraction failed, return original (might already be just method)
            return code
        except:
            return code
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean generated code to remove common artifacts"""
        if not code:
            return ""
        
        code = re.sub(r'```python\s*\n?', '', code)
        code = re.sub(r'```\s*$', '', code)
        code = re.sub(r'```\s*\n?', '', code)
        code = re.sub(r'^Here is the code:\s*\n?', '', code, flags=re.MULTILINE)
        code = re.sub(r'^Here\'s the code:\s*\n?', '', code, flags=re.MULTILINE)
        code = re.sub(r'^The code is:\s*\n?', '', code, flags=re.MULTILINE)
        code = re.sub(r'^Here is.*?:\s*\n?', '', code, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove any leading/trailing explanation text
        lines = code.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            if line.strip().startswith('def ') or line.strip().startswith('class '):
                in_code = True
            if in_code:
                code_lines.append(line)
            elif line.strip() and not any(word in line.lower() for word in ['here', 'code', 'function', 'generated']):
                in_code = True
                code_lines.append(line)
        
        return '\n'.join(code_lines).strip()
    
    def _create_simplified_prompt(self, specification: Dict[str, Any], is_class_method: bool) -> str:
        """Create a simplified prompt for retry attempts"""
        # Extract only essential information
        func_name = specification.get('function_name', 'function')
        english_summary = specification.get('english_summary', '')
        return_type = specification.get('return_type', '')
        
        # Get signature info
        signature = specification.get('signature', {})
        params = signature.get('parameters', [])
        param_names = [p.get('name', '') for p in params if isinstance(p, dict)][:5]
        param_str = ', '.join(param_names) if param_names else '...'
        
        if is_class_method:
            return f"""Generate ONLY the method code for: def {func_name}(self, {param_str}):

Summary: {english_summary[:200]}
Return type: {return_type}

CRITICAL SYNTAX REQUIREMENTS:
    - Use 4 spaces for indentation (NO tabs)
- All control structures MUST end with ':'
- Properly close all brackets/parentheses
- Valid Python syntax that can be parsed by AST

Generate only the method definition starting with 'def {func_name}(self, {param_str}):'
No explanations, no markdown, just valid Python code."""
        else:
            return f"""Generate Python function code:

Function name: {func_name}
Parameters: {param_str}
Summary: {english_summary[:200]}
Return type: {return_type}

CRITICAL SYNTAX REQUIREMENTS:
    - Use 4 spaces for indentation (NO tabs)
- All control structures MUST end with ':'
- Properly close all brackets/parentheses
- Valid Python syntax that can be parsed by AST

Generate only the function code starting with 'def {func_name}({param_str}):'
No explanations, no markdown, just valid Python code."""
    
    def _create_ultra_simple_prompt(self, specification: Dict[str, Any], is_class_method: bool) -> str:
        """Create ultra-simple prompt as last resort"""
        func_name = specification.get('function_name', 'function')
        signature = specification.get('signature', {})
        params = signature.get('parameters', [])
        param_names = [p.get('name', '') for p in params if isinstance(p, dict)][:5]
        param_str = ', '.join(param_names) if param_names else ''
        return_type = specification.get('return_type', 'Any')
        english_summary = specification.get('english_summary', '')[:150]
        
        if is_class_method:
            return f"""Generate Python method code:

def {func_name}(self, {param_str}) -> {return_type}:
    \"\"\"{english_summary}\"\"\"
    # Implement the function logic here
    pass

CRITICAL: Return ONLY valid Python code, no explanations."""
        else:
            return f"""Generate Python function code:

def {func_name}({param_str}) -> {return_type}:
    \"\"\"{english_summary}\"\"\"
    # Implement the function logic here
    pass

CRITICAL: Return ONLY valid Python code, no explanations."""


class TestGenerationNode(BaseNode):
    """Loads user-provided tests for behavioral equivalence validation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from utils.test_loader import TestLoader
        from utils.call_llm import call_llm

        self.test_loader = TestLoader()
        # Used by _generate_tests when LLM-generated tests are enabled; system prompt enforces JSON harness shape.
        self.call_llm = partial(call_llm, config=self.config, system=SYSTEM_TEST_JSON)
        self.few_shot_enhancer = FewShotPromptEnhancer()
    
    def _merge_and_deduplicate_cases(
        self, cases: List[Dict[str, Any]], max_keep: int
    ) -> List[Dict[str, Any]]:
        seen: Set[str] = set()
        out: List[Dict[str, Any]] = []
        for t in cases:
            if not isinstance(t, dict):
                continue
            try:
                sig = (
                    str(t.get("test_name"))
                    + "|"
                    + json.dumps(t.get("inputs") or {}, sort_keys=True, default=str)
                )
            except (TypeError, ValueError):
                sig = str(t)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(t)
            if len(out) >= max_keep:
                break
        return out

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Load repo tests where present; synthesize behavioral harness tests when needed."""
        print("    Loading / synthesizing behavioral tests...")

        specifications = context.get("specifications", {})
        generated_tests = context.setdefault("generated_tests", {})
        project_path = context.get("project_path", "")

        llm_fallback = bool(self.config.get("enable_llm_generated_tests", True))
        llm_topup = bool(self.config.get("llm_generated_tests_when_insufficient", True))
        min_cases = max(1, int(self.config.get("min_behavioral_cases", 3)))
        max_llm_rounds = max(1, int(self.config.get("max_llm_test_generation_rounds_per_func", 2)))
        max_keep = max(min_cases + 6, int(self.config.get("max_generated_tests_cap", 28)))
        harness_mode_cfg = str(self.config.get("harness_mode", "auto")).lower()

        for func_id, spec_data in specifications.items():
            if not spec_data.get("success", False):
                continue

            file_path = spec_data.get("file_path", "")
            function_name = spec_data["function_name"]
            prev = generated_tests.get(func_id)
            prev_tests: List[Any] = (prev.get("tests") if isinstance(prev, dict) else None) or []
            llm_rounds = (
                int(prev.get("llm_generation_rounds", 0))
                if isinstance(prev, dict)
                else 0
            )

            print(f"      Tests for {function_name}...")
            unittest_cases = []
            try:
                unittest_cases = self.test_loader.load_tests_for_function(
                    file_path, function_name, project_path
                )
            except Exception as e:
                print(f"        WARNING: loader error: {e}")

            pytest_paths: List[str] = []
            if harness_mode_cfg in ("auto", "pytest"):
                try:
                    from utils.pytest_harness import discover_pytest_paths

                    pytest_paths = discover_pytest_paths(file_path, project_path)
                except Exception as e:
                    print(f"        WARNING: pytest discovery error: {e}")

            merged: List[Dict[str, Any]] = []
            if unittest_cases:
                merged = [u for u in unittest_cases if isinstance(u, dict)]
            elif isinstance(prev_tests, list) and isinstance(prev, dict) and prev.get("harness_mode") != "pytest":
                merged = [t for t in prev_tests if isinstance(t, dict)]

            qualified_key = (
                func_id.split("::", 1)[1] if "::" in func_id else function_name
            )
            use_pytest = False
            if harness_mode_cfg == "pytest":
                use_pytest = bool(pytest_paths)
            elif harness_mode_cfg == "auto":
                use_pytest = len(merged) == 0 and bool(pytest_paths)

            tr = context.get("test_results", {}).get(func_id)
            feedback = (
                None
                if not tr
                else {
                    "missing_branches": tr.get("missing_branches") or [],
                    "missing_lines": tr.get("missing_lines") or [],
                    "branch_coverage": tr.get("branch_coverage", 0.0),
                    "metrics": {"branch_coverage": tr.get("branch_coverage", 0.0)},
                }
            )

            orig = context.get("original_code", {}).get(func_id, "") or ""
            need_llm = (
                not use_pytest
                and llm_fallback
                and orig.strip()
                and llm_rounds < max_llm_rounds
                and (len(merged) == 0 or (llm_topup and len(merged) < min_cases))
            )
            if need_llm:
                spec = spec_data.get("specification") or {}
                gen = self._generate_tests(orig, spec, function_name, feedback=feedback)
                if gen:
                    llm_rounds += 1
                for t in gen:
                    if isinstance(t, dict):
                        merged.append(t)

            merged = self._merge_and_deduplicate_cases(merged, max_keep)

            ulen = len([u for u in (unittest_cases or []) if isinstance(u, dict)])

            if use_pytest:
                generated_tests[func_id] = {
                    "tests": [],
                    "harness_mode": "pytest",
                    "pytest_paths": pytest_paths,
                    "qualified_key": qualified_key,
                    "function_name": function_name,
                    "file_path": file_path,
                    "llm_generation_rounds": llm_rounds,
                    "test_sources": {
                        "from_unittest_loader": ulen,
                        "from_pytest_harness": len(pytest_paths),
                        "total_cases": 0,
                    },
                }
                print(
                    f"        pytest harness: {len(pytest_paths)} module(s) "
                    f"(loader cases={ulen}, mode={harness_mode_cfg})"
                )
            else:
                generated_tests[func_id] = {
                    "tests": merged,
                    "function_name": function_name,
                    "file_path": file_path,
                    "llm_generation_rounds": llm_rounds,
                    "test_sources": {
                        "from_unittest_loader": ulen,
                        "total_cases": len(merged),
                    },
                }

                if ulen > 0:
                    print(f"        {len(merged)} case(s): {ulen} from repo tests (+ LLM synthesized as needed)")
                else:
                    print(f"        {len(merged)} harness case(s) (LLM + carried forward)")

        print(f"    Prepared behavioral tests for {len(generated_tests)} function(s)")
        return context
    
    def _generate_tests(
        self,
        original_code: str,
        specification: Dict[str, Any],
        function_name: str,
        feedback: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate tests for a function, using path analysis if available"""
        
        # Get coverage information from feedback
        missing_branches = feedback.get('missing_branches', []) if feedback else []
        missing_lines = feedback.get('missing_lines', []) if feedback else []
        current_coverage = feedback.get('branch_coverage', 0.0) if feedback else 0.0
        
        implementation_details = (
            specification.get('implementation')
            or specification.get('internal_logic')
            or specification.get('logic')
            or {}
        )
        if not isinstance(implementation_details, dict):
            implementation_details = {}
        signature_info = (
            specification.get('signature')
            or specification.get('function_signature')
            or {}
        )
        if not isinstance(signature_info, dict):
            signature_info = {}
        parameters = signature_info.get('parameters', [])
        
        required_parameters: List[Dict[str, Any]] = []
        for param in parameters or []:
            if not isinstance(param, dict):
                continue
            name = param.get('name')
            if not name or name == 'self':
                continue
            required_parameters.append(param)
        
        import_candidates: List[str] = []
        dependency_sources = []
        imports_field = specification.get('imports')
        if isinstance(imports_field, list):
            dependency_sources.append(imports_field)
        elif isinstance(imports_field, dict):
            items = imports_field.get('items', [])
            if isinstance(items, list):
                dependency_sources.append(items)
        
        impl_imports = implementation_details.get('imports', []) if isinstance(implementation_details, dict) else []
        if impl_imports:
            dependency_sources.append(impl_imports)
        
        deps_struct = specification.get('dependencies')
        if isinstance(deps_struct, dict):
            deps_imports = deps_struct.get('imports', [])
            if isinstance(deps_imports, list):
                dependency_sources.append(deps_imports)
        elif isinstance(deps_struct, list):
            dependency_sources.append(deps_struct)
        deps_and_imports = specification.get('dependencies_and_imports')
        if isinstance(deps_and_imports, dict):
            deps_ai_imports = deps_and_imports.get('imports', [])
            if isinstance(deps_ai_imports, list):
                dependency_sources.append(deps_ai_imports)
        
        def flatten_imports(source: Any) -> List[str]:
            """Recursively flatten nested lists of imports"""
            result = []
            if isinstance(source, str):
                result.append(source)
            elif isinstance(source, list):
                for item in source:
                    result.extend(flatten_imports(item))
            elif isinstance(source, dict):
                for key, value in source.items():
                    if isinstance(key, str) and key.startswith(('import ', 'from ')):
                        result.append(key)
                    result.extend(flatten_imports(value))
            return result
        
        for source in dependency_sources:
            flattened = flatten_imports(source)
            import_candidates.extend(item for item in flattened if isinstance(item, str) and item.strip())
        
        imports_unique = []
        seen_imports = set()
        for imp in import_candidates:
            normalized = imp.strip()
            if normalized and normalized not in seen_imports:
                imports_unique.append(normalized)
                seen_imports.add(normalized)
        
        dependencies_candidates: List[str] = []
        dependencies_sources = []
        deps_list = specification.get('dependencies')
        if isinstance(deps_list, list):
            dependencies_sources.append(deps_list)
        elif isinstance(deps_list, dict):
            dependencies_sources.append(deps_list.get('items', []))
        impl_deps = implementation_details.get('dependencies', []) if isinstance(implementation_details, dict) else []
        if impl_deps:
            dependencies_sources.append(impl_deps)
        if isinstance(deps_and_imports, dict):
            dependencies_sources.append(deps_and_imports.get('dependencies', []))
        
        for source in dependencies_sources:
            if isinstance(source, list):
                dependencies_candidates.extend(item for item in source if isinstance(item, str))
        
        dependencies_unique = []
        seen_deps = set()
        for dep in dependencies_candidates:
            normalized = dep.strip()
            if normalized and normalized not in seen_deps:
                dependencies_unique.append(normalized)
                seen_deps.add(normalized)
        
        class_ctx = specification.get('class_context', {})
        class_name = specification.get('class_name') or class_ctx.get('class_name')
        class_attributes = class_ctx.get('class_attributes', []) if isinstance(class_ctx, dict) else []
        
        side_effects = specification.get('side_effects')
        error_handling = specification.get('error_handling')
        edge_cases = specification.get('edge_cases')
        user_stories = specification.get('user_stories') or []
        success_criteria = specification.get('success_criteria') or []
        spec_test_matrix = specification.get('test_matrix') or []
        logical_plan = specification.get('logical_deletion') or {}
        coverage_metrics = feedback.get('metrics', {}) if isinstance(feedback, dict) else {}
        missing_lines_feedback = coverage_metrics.get('missing_lines') or []
        missing_branches_feedback = coverage_metrics.get('missing_branches') or []
        branch_coverage = coverage_metrics.get('branch_coverage')
        coverage_gaps: List[str] = []
        if missing_lines_feedback:
            coverage_gaps.append(f"Missing executable lines: {missing_lines_feedback}")
        if missing_branches_feedback:
            coverage_gaps.append(f"Missing branches: {missing_branches_feedback}")
        if isinstance(branch_coverage, (int, float)) and branch_coverage < 0.9999:
            coverage_gaps.append(f"Recorded branch coverage: {branch_coverage:.2%}")
        coverage_feedback_section = "\n".join(f"- {entry}" for entry in coverage_gaps) if coverage_gaps else "None reported."
        
        def _format_list(items: List[str]) -> str:
            if not items:
                return "- None"
            return "\n".join(f"- {item}" for item in items)
        
        imports_section = _format_list(imports_unique)
        dependencies_section = _format_list(dependencies_unique)
        
        # Handle edge_cases, error_handling, and side_effects which can be lists or dicts
        if isinstance(edge_cases, list):
            edge_cases_section = json.dumps(edge_cases, indent=2) if edge_cases else "No additional edge cases provided."
        elif isinstance(edge_cases, dict):
            edge_cases_section = json.dumps(edge_cases, indent=2) if edge_cases else "No additional edge cases provided."
        else:
            edge_cases_section = "No additional edge cases provided."
        
        if isinstance(error_handling, dict):
            error_handling_section = json.dumps(error_handling, indent=2) if error_handling else "No explicit error handling documented."
        elif isinstance(error_handling, list):
            error_handling_section = json.dumps(error_handling, indent=2) if error_handling else "No explicit error handling documented."
        else:
            error_handling_section = "No explicit error handling documented."
        
        if isinstance(side_effects, list):
            side_effects_section = json.dumps(side_effects, indent=2) if side_effects else "No side effects documented."
        elif isinstance(side_effects, dict):
            side_effects_section = json.dumps(side_effects, indent=2) if side_effects else "No side effects documented."
        else:
            side_effects_section = "No side effects documented."
        english_summary = specification.get('english_summary') or ""
        english_summary_section = english_summary if english_summary else "Not provided."
        
        drift_issues = specification.get('drift_issues') or []
        if drift_issues:
            drift_section = "\n".join(f"- {issue}" for issue in drift_issues)
        else:
            drift_section = "None detected."
        parameters_section = []
        for param in required_parameters:
            description = param.get('description') or "No description provided."
            param_type = param.get('type') or "Any"
            parameters_section.append(f"- `{param['name']}` ({param_type}): {description}")
        if parameters_section:
            parameters_text = "\n".join(parameters_section)
        else:
            # For parameterless functions, provide clear guidance
            if class_name:
                parameters_text = "- This method takes no parameters beyond `self`. The `inputs` object should be an empty dict `{}`."
            else:
                parameters_text = "- This function takes no parameters. The `inputs` object should be an empty dict `{}`."
        
        class_context_section = "No class context."
        if class_name:
            attributes_text = ", ".join(class_attributes) if class_attributes else "No persistent attributes documented."
            class_context_section = (
                f"The callable is a method of the `{class_name}` class. "
                f"Documented persistent attributes: {attributes_text}. "
                "The harness instantiates this class and binds the method before execution, so do NOT include `self` in the `inputs` payload."
            )

        bounded_section = ""
        bcb = specification.get("bounded_context_bundle") or ""
        if isinstance(bcb, str) and bcb.strip():
            from agents.context_bundle import truncate_utf8

            cap = int(self.config.get("context_test_prompt_bundle_bytes", 12_288))
            clip = truncate_utf8(bcb.strip(), cap)
            bounded_section = (
                "\n### Bounded dependency context (read-only excerpts; use for realism only)\n"
                "```text\n"
                + clip
                + "\n```\n"
            )
        
        def _format_user_story(story: Dict[str, Any]) -> str:
            if not isinstance(story, dict):
                return ""
            story_id = story.get('id', '')
            priority = story.get('priority', '')
            title = story.get('title') or story.get('narrative') or "Untitled story"
            narrative = story.get('narrative') or story.get('summary') or ""
            acceptance = story.get('acceptance') or []
            acceptance_lines = "\n".join(
                f"    - Given {acc.get('given', '')}, When {acc.get('when', '')}, Then {acc.get('then', '')}"
                for acc in acceptance if isinstance(acc, dict)
            )
            if not acceptance_lines:
                acceptance_lines = "    - None documented"
            return (
                f"- [{priority}] {story_id} {title}\n"
                f"  Narrative: {narrative}\n"
                f"  Acceptance:\n{acceptance_lines}"
            )
        
        # Ensure user_stories and success_criteria are lists of dicts
        safe_user_stories = []
        for story in user_stories:
            if isinstance(story, dict):
                safe_user_stories.append(story)
            elif isinstance(story, str):
                # Convert string to dict format
                safe_user_stories.append({'id': '', 'title': story, 'narrative': story})
        
        safe_success_criteria = []
        for criterion in success_criteria:
            if isinstance(criterion, dict):
                safe_success_criteria.append(criterion)
            elif isinstance(criterion, str):
                # Convert string to dict format
                safe_success_criteria.append({'id': '', 'description': criterion})
        
        user_story_section = "\n".join(filter(None, (_format_user_story(story) for story in safe_user_stories))) or "None documented."
        success_criteria_section = "\n".join(
            f"- {criterion.get('id', '')}: {criterion.get('description', '')} | Metric: {criterion.get('metric', '')} Target: {criterion.get('target', '')}"
            for criterion in safe_success_criteria
        ) or "None documented."
        test_matrix_section = json.dumps(spec_test_matrix, indent=2) if spec_test_matrix else "[]"
        
        # Incorporate path analysis if available
        path_analysis = specification.get('path_analysis', {})
        path_section = ""
        if path_analysis.get('paths'):
            paths = path_analysis['paths']
            path_section = f"\n\n### Execution Path Analysis\n"
            path_section += f"This function has {len(paths)} distinct execution paths identified through AST analysis:\n"
            for i, path in enumerate(paths[:8], 1):
                path_type = path.get('type', 'unknown')
                conditions = path.get('conditions', [])
                path_section += f"{i}. Path {i} ({path_type}): Conditions: {', '.join(conditions[:2]) if conditions else 'default'}\n"
            path_section += "\nGenerate test cases that exercise EACH of these paths. Map each test to a specific path via story_refs.\n"
        
        call_instructions = []
        if class_name:
            call_instructions.append(
                "- Rely on the harness-provided instance; only include arguments after `self` in the `inputs` object."
            )
            call_instructions.append(
                "- When the method updates instance attributes, include `state_assertions` documenting the expected attribute values after the call."
            )
        if not required_parameters:
            # Special handling for parameterless functions
            call_instructions.append(
                "- IMPORTANT: This function takes no parameters. Use `\"inputs\": {}` (empty dict) for all test cases."
            )
            if class_name:
                call_instructions.append(
                    "- For class methods with no parameters, test different instance states (before/after method calls)."
                )
            else:
                call_instructions.append(
                    "- For parameterless functions, test different scenarios based on function behavior (return values, side effects, etc.)."
                )
        call_instructions.append("- Cover every branch, raise path, and edge case described in the specification.")
        if missing_lines_feedback:
            call_instructions.append(f"CRITICAL: Generate tests that execute lines {missing_lines_feedback}. These lines were not covered in previous iterations.")
            call_instructions.append(f"- For each missing line, create a test case with inputs that force execution through that specific line.")
        if missing_branches_feedback:
            call_instructions.append("CRITICAL: Generate tests that exercise each previously missing branch.")
            call_instructions.append("- Analyze the control flow conditions (if/else, loops, exceptions) and provide inputs that trigger each branch.")
        if isinstance(branch_coverage, (int, float)) and branch_coverage < 0.8:
            call_instructions.append(f"Current branch coverage is only {branch_coverage:.1%}. Generate additional tests to increase coverage to at least 80%.")
            call_instructions.append("- Focus on boundary conditions, edge cases, and error paths that may not be covered.")
        
        # Add path-specific test generation guidance
        if path_analysis.get('paths'):
            paths = path_analysis['paths']
            call_instructions.append(f"\nPATH-SPECIFIC TESTING REQUIREMENTS:")
            call_instructions.append(f"- Generate at least one test case for each of the {len(paths)} identified execution paths.")
            call_instructions.append("- For each path, create inputs that satisfy the path's conditions to force execution through that specific path.")
            for i, path in enumerate(paths[:5], 1):
                path_type = path.get('type', 'unknown')
                conditions = path.get('conditions', [])
                if conditions:
                    call_instructions.append(f"  Path {i} ({path_type}): Test with inputs satisfying: {', '.join(conditions[:2])}")
        
        if logical_plan.get('slice_assertions'):
            call_instructions.append("- For each slice assertion below, craft at least one test whose inputs trigger the precondition and verify the documented effect.")
        call_instructions.append("- Use descriptive `test_name` values indicating the specific scenario being validated.")
        call_instructions.append("- Ensure expected outputs match the exact return value or exception semantics.")
        if required_parameters:
            call_instructions.insert(0, "- Every test case must provide all required parameters listed above.")
        else:
            call_instructions.insert(0, "- IMPORTANT: This function has no parameters. Use `\"inputs\": {}` for all test cases.")
        call_instructions_text = "\n".join(call_instructions)
        
        # Determine inputs example and instruction for prompt
        is_parameterless = not required_parameters
        inputs_example = "{}" if is_parameterless else '{"arg1": value1, "arg2": value2}'
        inputs_instruction = "Use `\"inputs\": {}` (empty dict) for all test cases." if is_parameterless else "Provide all required parameters in the `inputs` object."
        
        internal_logic = specification.get('internal_logic')
        if isinstance(internal_logic, list):
            flow_section = "\n".join(f"- {item}" for item in internal_logic if item)
        elif isinstance(internal_logic, str):
            flow_section = f"- {internal_logic}"
        else:
            flow_section = "None documented."
        
        logical_requirements_section = "No logical deletion constraints documented."
        if logical_plan:
            summary = logical_plan.get('summary', '')
            critical = logical_plan.get('critical_lines', [])[:12]
            deletable = logical_plan.get('deletable_lines', [])[:12]
            assertions = logical_plan.get('slice_assertions', [])
            assertion_lines = []
            for assertion in assertions[:6]:
                assertion_lines.append(
                    f"- {assertion['assertion_id']}: When {assertion['precondition']} expect {assertion['expected_effect']}"
                )
            assertion_text = "\n".join(assertion_lines) if assertion_lines else "- None"
            logical_requirements_section = (
                f"Critical lines to preserve: {critical}\n"
                f"Lines targeted for deletion (ensure tests fail if reintroduced): {deletable}\n"
                f"{summary}\n"
                f"Slice assertions to enforce:\n{assertion_text}"
            )
        
        prompt = f"""
Generate comprehensive unit tests for the following Python callable. Use the contextual information to ensure every documented behavior, dependency, and edge case is exercised.

### Imports that must be available
{imports_section}

### Dependencies and helper calls to exercise
{dependencies_section}

### Class or stateful context
{class_context_section}
{bounded_section}

### User stories and acceptance scenarios
{user_story_section}

### Success criteria
{success_criteria_section}

### Required parameters
{parameters_text}

### Documented side effects
{side_effects_section}

### Documented error handling
{error_handling_section}

### Documented edge cases
{edge_cases_section}

### Original implementation
```python
{original_code}
```

### Specification (reference)
{json.dumps(specification, indent=2)}

### English summary
{english_summary_section}

### Known spec-code drift warnings
{drift_section}

### Flow checkpoints from specification
{flow_section}

### Coverage feedback from prior iteration
{coverage_feedback_section}

### Suggested test matrix from specification
{test_matrix_section}
### Logical deletion + slice assertions
{logical_requirements_section}
{path_section}
### Test design requirements
{call_instructions_text}

### Output format
Return a JSON array of test cases with this structure:
[
  {{
    "test_name": "test_description",
    "inputs": {inputs_example},
    "expected_output": expected_value,
    "expected_exception": null,
    "description": "What this test validates",
    "state_assertions": []
  }}
]

CRITICAL: {inputs_instruction}

Generate at least 10-12 varied test cases covering:
1. Normal behavior (typical inputs that exercise main logic)
2. Edge cases (empty inputs: empty lists, empty dicts, empty strings, zero, None, boundary values)
3. Error paths (invalid inputs, exceptions, type errors)
4. Stateful scenarios (for class methods - state changes, multiple calls)
5. All branch paths (every if/else branch, every loop path)
6. Return value variations (different return types/paths, None returns)
7. Boundary conditions (min/max values, empty collections, single items)
8. Type variations (if function handles multiple types)

For each test case, ensure:
- Inputs match EXACT parameter names from specification (not variations)
- Expected outputs are precise and match specification exactly
- Edge cases test ALL boundaries (empty lists, zero, negative values, None, empty strings)
- Error cases test ALL exception types mentioned in specification
- State assertions verify ALL instance attribute changes (for class methods)
- Test cases cover ALL control flow paths (every if branch, every loop iteration)
- Include tests for None/NoneType inputs if function accepts optional parameters

Return ONLY the JSON array, no other text.
"""
        
        # Enhance prompt with coverage targets if coverage is low
        if current_coverage < 0.8 and (missing_branches or missing_lines):
            from agents.enhanced_test_generation import EnhancedTestGenerator
            enhanced_gen = EnhancedTestGenerator()
            prompt = enhanced_gen.enhance_test_prompt_with_coverage_targets(
                prompt, missing_branches, missing_lines, current_coverage
            )
        
        # Add few-shot examples for test generation
        prompt = self.few_shot_enhancer.add_test_examples_to_prompt(prompt, specification, function_name)
        
        primary_response = ""
        try:
            primary_response = self.call_llm(prompt)
        except Exception:
            primary_response = ""
        
        tests = self._parse_tests_response(primary_response, required_parameters, specification)
        if tests:
            return tests
        
        fallback_prompt = self._build_fallback_test_prompt(
            function_name=function_name,
            english_summary_section=english_summary_section,
            parameters_text=parameters_text,
            coverage_feedback_section=coverage_feedback_section,
            test_matrix_section=test_matrix_section,
            flow_section=flow_section,
            original_code=original_code,
            user_story_section=user_story_section
        )
        
        fallback_response = ""
        try:
            fallback_response = self.call_llm(fallback_prompt)
        except Exception:
            fallback_response = ""
        
        return self._parse_tests_response(fallback_response, required_parameters, specification)
    
    def _parse_tests_response(
        self,
        response: str,
        required_parameters: List[Dict[str, Any]],
        specification: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        if not response:
            return []
        
        # Try multiple JSON extraction strategies
        json_str = None
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
        else:
            # Try to find JSON after "```json" or "```"
            code_block_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
            if code_block_match:
                json_str = code_block_match.group(1)
            else:
                # Try to find any array-like structure
                array_match = re.search(r'(\[[\s\S]{50,}\])', response)
                if array_match:
                    json_str = array_match.group(1)
        
        if not json_str:
            return []
                
        try:
            tests = json.loads(json_str)
        except Exception:
            return []
        return self._sanitize_tests(tests, required_parameters, specification)
    
    def _build_fallback_test_prompt(
        self,
        function_name: str,
        english_summary_section: str,
        parameters_text: str,
        coverage_feedback_section: str,
        test_matrix_section: str,
        flow_section: str,
        original_code: str,
        user_story_section: str
    ) -> str:
        # Determine if function is parameterless
        is_parameterless = "no parameters" in parameters_text.lower() or "empty dict" in parameters_text.lower()
        
        inputs_example = "{}" if is_parameterless else '{"param": value}'
        inputs_instruction = "Use `\"inputs\": {}` (empty dict) for all test cases." if is_parameterless else "Provide all required parameters in the `inputs` object."
        
        return f"""
The previous response failed to produce valid JSON tests. Return ONLY a JSON array of test cases using this exact schema:
[
  {{
    "test_name": "test_description",
    "inputs": {inputs_example},
    "expected_output": expected_value,
    "expected_exception": null,
    "description": "What this test validates",
    "state_assertions": []
  }}
]

Function: {function_name}
Summary: {english_summary_section}
Required parameters:
    {parameters_text}
    
IMPORTANT: {inputs_instruction}

Coverage feedback:
    {coverage_feedback_section}

Relevant user stories:
    {user_story_section}

Flow checkpoints:
    {flow_section}

Suggested scenarios:
    {test_matrix_section}

Implementation reference:
```python
{original_code}
```

Return at least 6 diverse test cases. For parameterless functions, test different scenarios based on the function's behavior (return values, side effects, state changes).
Return ONLY the JSON array, no markdown, no explanations.
"""
    
    def _sanitize_tests(self, tests: Any, required_parameters: List[Dict[str, Any]], specification: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate and normalize generated tests to match the execution harness requirements."""
        if not isinstance(tests, list):
            return []
        
        required_param_names = [param['name'] for param in required_parameters if 'name' in param]

        # Build fallback values from specification parameter metadata
        parameter_examples: Dict[str, Any] = {}
        signature = specification.get('signature', {})
        for param in signature.get('parameters', []) or []:
            if not isinstance(param, dict):
                continue
            name = param.get('name')
            if not name or name == 'self':
                continue
            examples = param.get('example_values') or param.get('examples')
            if isinstance(examples, list) and examples:
                parameter_examples[name] = examples[0]
            else:
                guessed = self._guess_default_value(param)
                if guessed is not None:
                    parameter_examples[name] = guessed
        
        sanitized: List[Dict[str, Any]] = []
        has_class_context = bool(specification.get('class_context') or specification.get('class_name'))
        
        for idx, test in enumerate(tests):
            if not isinstance(test, dict):
                continue
            
            inputs = test.get('inputs')
            if not isinstance(inputs, dict):
                continue
            
            # For parameterless functions, ensure inputs is empty dict or handle gracefully
            if not required_param_names:
                # Parameterless function - inputs should be empty dict or we'll make it empty
                if not isinstance(inputs, dict):
                    inputs = {}
                # Don't skip tests for parameterless functions
            else:
                # Function with parameters - validate all are present
                missing_params = [name for name in required_param_names if name not in inputs or inputs[name] is None]
                for missing in missing_params:
                    if missing in parameter_examples:
                        inputs[missing] = parameter_examples[missing]
                
                if any(name not in inputs or inputs[name] is None for name in required_param_names):
                    continue
            
            expected_output = test.get('expected_output')
            expected_exception = test.get('expected_exception')
            if expected_output is None and not expected_exception:
                continue
            
            sanitized_test = {
                'test_name': test.get('test_name') or f"auto_test_{idx + 1}",
                'inputs': inputs,
                'expected_output': expected_output,
                'expected_exception': expected_exception,
                'description': test.get('description', '')
            }
            sanitized.append(sanitized_test)
        
        return sanitized
    
    def _guess_default_value(self, param: Dict[str, Any]) -> Any:
        """Guess a reasonable default value when examples are unavailable."""
        param_type = (param.get('type') or "").lower()
        name = (param.get('name') or "").lower()
        
        if 'int' in param_type and 'float' not in param_type:
            return 1
        if 'float' in param_type or 'double' in param_type:
            return 1.0
        if 'bool' in param_type:
            return True
        if 'str' in param_type or 'string' in param_type:
            return "example"
        if 'list' in param_type or 'sequence' in param_type:
            return []
        if 'dict' in param_type or 'mapping' in param_type:
            return {}
        if 'tuple' in param_type or 'set' in param_type:
            return []
        if 'callable' in param_type or 'function' in param_type:
            return None
        if name.endswith('path') or 'path' in name:
            return "/tmp/example"
        if any(keyword in name for keyword in ('count', 'size', 'length', 'limit')):
            return 1
        return None


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
            
            # Re-run tests whenever execution is invoked — regenerated code changes across
            # iterations, failure-driven refinement, and hybrid loops. Skipping stale
            # results left behavioral_test_similarity frozen at empty / zero forever.
            print(f"      Testing {specifications[func_id]['function_name']}...")
            
            try:
                # Get original code from context (NOT from specification)
                original_code = context.get('original_code', {}).get(func_id, '')
                if not original_code:
                    # Fallback: try to get from func_info if available
                    func_info = context.get('function_info', {}).get(func_id, {})
                    original_code = func_info.get('source_code', '')

                gen_entry = generated_tests[func_id]
                if gen_entry.get("harness_mode") == "pytest":
                    results = self._execute_pytest_harness(
                        context,
                        func_id,
                        specifications[func_id],
                        original_code,
                        regenerated_code[func_id]["code"],
                        gen_entry,
                    )
                else:
                    results = self._execute_tests(
                        original_code,
                        regenerated_code[func_id]['code'],
                        gen_entry['tests'],
                        specifications[func_id]['function_name'],
                        specifications[func_id].get('imports', []),
                        specifications[func_id]['specification'],
                        specifications[func_id]['file_path']
                    )
                
                test_results[func_id] = results
                
                passed = results['original_passed'] + results['regenerated_passed']
                total = results['total_tests'] * 2
                print(f"        Tests passed: {passed}/{total}")
                
            except Exception as e:
                traceback.print_exc()
                print(f"        ERROR: {e}")
                continue
        
        context['test_results'] = test_results
        print(f"    Executed tests for {len(test_results)} functions")
        
        return context

    def _execute_pytest_harness(
        self,
        context: Dict[str, Any],
        func_id: str,
        spec_data: Dict[str, Any],
        original_code: str,
        regenerated_code: str,
        gen_entry: Dict[str, Any],
    ) -> Dict[str, Any]:
        from utils.pytest_harness import run_pytest_harness_for_function

        project_path = context.get("project_path", "")
        file_path = spec_data.get("file_path") or gen_entry.get("file_path", "")
        qualified_key = gen_entry.get("qualified_key") or (
            func_id.split("::", 1)[1] if "::" in func_id else spec_data.get("function_name", "")
        )
        pytest_paths = gen_entry.get("pytest_paths") or []
        timeout_sec = int(self.config.get("pytest_harness_timeout_sec", 300))
        cache = context.setdefault("pytest_baseline_cache", {})

        return run_pytest_harness_for_function(
            project_path=project_path,
            source_file=file_path,
            qualified_key=qualified_key,
            pytest_paths=pytest_paths,
            original_source=original_code,
            regenerated_source=regenerated_code,
            baseline_cache=cache,
            timeout_sec=timeout_sec,
        )
    
    def _execute_tests(self, original_code: str, regenerated_code: str, tests: List[Dict[str, Any]], 
                       function_name: str, imports: List[str], specification: Dict[str, Any],
                       file_path: str) -> Dict[str, Any]:
        """Execute tests on both code versions with branch coverage tracking"""
        
        valid_tests: List[Dict[str, Any]] = [
            t for t in tests
            if isinstance(t, dict) and isinstance(t.get('inputs'), dict)
        ]
        
        results = {
            'total_tests': len(valid_tests),
            'original_passed': 0,
            'original_failed': 0,
            'regenerated_passed': 0,
            'regenerated_failed': 0,
            'failures': [],
            'behavioral_match': True,  # Set False below when no tests
            'branch_coverage': 0.0,
            'coverage_complete': False
        }
        
        if not valid_tests:
            results['behavioral_match'] = False  # No tests = no evidence; default to 0
            return results
        
        # Create temporary files FIRST, before starting coverage
        original_module_path = self._create_temp_module_file(
            original_code,
            imports,
            prefix="original",
            specification=specification,
            file_path=file_path,
            is_regenerated=False
        )
        regenerated_module_path = self._create_temp_module_file(
            regenerated_code,
            imports,
            prefix="regenerated",
            specification=specification,
            file_path=file_path,
            is_regenerated=True
        )
        
        # Start coverage - it will automatically track imported files
        coverage_original = Coverage(branch=True, data_file=None)
        coverage_original.erase()
        coverage_original.start()
        
        coverage_regenerated = Coverage(branch=True, data_file=None)
        coverage_regenerated.erase()
        coverage_regenerated.start()
        
        with ExitStack() as stack:
            stack.callback(self._cleanup_temp_module, original_module_path)
            stack.callback(self._cleanup_temp_module, regenerated_module_path)
            
            return self._run_tests_loop(
                tests=valid_tests,
                original_module_path=original_module_path,
                regenerated_module_path=regenerated_module_path,
                function_name=function_name,
                coverage_original=coverage_original,
                coverage_regenerated=coverage_regenerated,
                results=results,
                original_code=original_code
            )
    
    def _run_single_test(self, module_path: str, function_name: str, test: Dict[str, Any], 
                         coverage_tracker: Coverage) -> Dict[str, Any]:
        """Run a single test case against a temporary module under coverage tracking"""
        # Use a consistent module name based on file path to avoid import caching issues
        # But ensure it's unique enough to avoid conflicts
        module_name = f"_temp_module_{os.path.basename(module_path).replace('.py', '').replace('-', '_')}"
        
        # Remove from sys.modules if it exists to force re-import
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        
        # Coverage is already started and will automatically track imported files
        
        try:
            spec.loader.exec_module(module)
            
            # Get instance for class methods
            instance = None
            if hasattr(module, "_get_last_instance"):
                try:
                    instance = module._get_last_instance()
                except Exception:
                    pass
            if instance is None and hasattr(module, "_last_instance"):
                instance = getattr(module, "_last_instance", None)
            
            # Prepare instance if needed
            if hasattr(module, "_prepare_instance"):
                try:
                    module._prepare_instance()
                    if instance is None:
                        instance = getattr(module, "_last_instance", None)
                except Exception:
                    pass
            
            inputs = test.get('inputs')
            if not isinstance(inputs, dict):
                raise ValueError("Invalid test inputs")
            
            # Process inputs: evaluate lambda functions and other callables
            processed_inputs = {}
            for key, value in inputs.items():
                if isinstance(value, str) and value.strip().startswith("lambda"):
                    # Try to evaluate lambda function string
                    try:
                        processed_inputs[key] = eval(value)
                    except Exception:
                        # If evaluation fails, keep as string (might be handled by function)
                        processed_inputs[key] = value
                else:
                    processed_inputs[key] = value
            
            call_inputs, state_overrides = self._separate_self_overrides(processed_inputs)
            
            # For class methods, ensure instance exists and has correct state
            if state_overrides:
                if hasattr(module, "_reset_instance"):
                    # Reset instance with state overrides
                    try:
                        instance = module._reset_instance(state_overrides)
                    except Exception:
                        pass
                elif hasattr(module, "_apply_state_overrides"):
                    # Apply state overrides to existing instance
                    try:
                        module._apply_state_overrides(state_overrides)
                        if hasattr(module, "_get_last_instance"):
                            try:
                                instance = module._get_last_instance()
                            except Exception:
                                pass
                        if instance is None and hasattr(module, "_last_instance"):
                            instance = getattr(module, "_last_instance", None)
                    except Exception:
                        pass
            
            # Get the callable - for class methods, call on instance
            target_callable = getattr(module, function_name)
            
            # Check if this is a class method by checking if instance exists and method needs self
            if instance is not None:
                # Ensure instance is refreshed
                if hasattr(module, "_get_last_instance"):
                    try:
                        instance = module._get_last_instance()
                    except Exception:
                        pass
                if instance is None and hasattr(module, "_last_instance"):
                    instance = getattr(module, "_last_instance", None)
                
                if instance is not None:
                    # Call method on instance
                    method = getattr(instance, function_name, None)
                    if method:
                        result = method(**call_inputs)
                    else:
                        # Fallback: try unbound method
                        result = target_callable(instance, **call_inputs)
                else:
                    # No instance available, try regular call
                    result = target_callable(**call_inputs)
            else:
                # Regular function call
                result = target_callable(**call_inputs)
            expected = test.get('expected_output')
            
            snapshot = self._capture_instance_state(module)
            
            # Deep comparison for complex types (dicts, lists, sets)
            passed = self._deep_equals(result, expected)
            
            return {
                'passed': passed,
                'output': result,
                'expected': expected,
                'state_snapshot': snapshot,
                'error': None
            }
            
        except Exception as exc:
            expected_exception = test.get('expected_exception')
            
            snapshot = self._capture_instance_state(module)
            
            if expected_exception and type(exc).__name__ == expected_exception:
                return {
                    'passed': True,
                    'output': f"Exception: {type(exc).__name__}",
                    'expected': f"Exception: {expected_exception}",
                    'state_snapshot': snapshot,
                    'error': None
                }
            
            return {
                'passed': False,
                'output': None,
                'expected': test.get('expected_output'),
                'error': str(exc),
                'state_snapshot': snapshot
            }
        
        finally:
            sys.modules.pop(module_name, None)
    
    def _run_tests_loop(
        self,
        tests: List[Dict[str, Any]],
        original_module_path: str,
        regenerated_module_path: str,
        function_name: str,
        coverage_original: Coverage,
        coverage_regenerated: Coverage,
        results: Dict[str, Any],
        original_code: str
    ) -> Dict[str, Any]:
        """Execute each generated test and aggregate coverage/statistics."""
        for test in tests:
            original_result = self._run_single_test(
                module_path=original_module_path,
                function_name=function_name,
                test=test,
                coverage_tracker=coverage_original
            )
            
            regenerated_result = self._run_single_test(
                module_path=regenerated_module_path,
                function_name=function_name,
                test=test,
                coverage_tracker=coverage_regenerated
            )
            
            if original_result['passed']:
                results['original_passed'] += 1
            else:
                results['original_failed'] += 1
            
            if regenerated_result['passed']:
                results['regenerated_passed'] += 1
            else:
                results['regenerated_failed'] += 1
            
            # Use deep equality for output comparison
            outputs_match = self._deep_equals(
                original_result['output'],
                regenerated_result['output']
            )
            original_state = original_result.get('state_snapshot')
            regenerated_state = regenerated_result.get('state_snapshot')
            state_status = self._deep_equals(original_state, regenerated_state) if original_state is not None or regenerated_state is not None else True
            
            if not (outputs_match and state_status):
                results['behavioral_match'] = False
                results['failures'].append({
                    'test': test,
                    'original_output': original_result['output'],
                    'regenerated_output': regenerated_result['output'],
                    'original_error': original_result.get('error'),
                    'regenerated_error': regenerated_result.get('error'),
                    'original_state': original_state,
                    'regenerated_state': regenerated_state
                })
        
        # Stop coverage tracking AFTER all tests complete
        try:
            coverage_original.stop()
            coverage_original.save()
        except Exception:
            pass
        
        # Try to calculate branch coverage, but don't fail if it doesn't work
        # Branch coverage with dynamic imports is unreliable, so we focus on test results
        try:
            coverage_info = self._calculate_branch_coverage(
                coverage_original,
                original_module_path,
                original_code
            )
            if not isinstance(coverage_info, dict):
                coverage_info = {
                    'coverage': 0.0,
                    'coverage_complete': False,
                    'missing_branches': [],
                    'missing_lines': []
                }
        except Exception:
            # If coverage calculation fails, use defaults
            coverage_info = {
                'coverage': 0.0,
                'coverage_complete': False,
                'missing_branches': [],
                'missing_lines': []
            }
        
        results['branch_coverage'] = coverage_info['coverage']
        results['coverage_complete'] = coverage_info['coverage_complete']
        results['missing_branches'] = coverage_info['missing_branches']
        results['missing_lines'] = coverage_info['missing_lines']
        return results
    
    def _capture_instance_state(self, module: Any) -> Optional[Dict[str, Any]]:
        """Capture a shallow, serializable snapshot of the bound instance state."""
        instance = None
        if hasattr(module, "_get_last_instance"):
            try:
                instance = module._get_last_instance()
            except Exception:
                instance = None
        if instance is None and hasattr(module, "_last_instance"):
            instance = getattr(module, "_last_instance", None)
        
        if instance is None:
            return None
        
        try:
            state = {}
            for key, value in getattr(instance, "__dict__", {}).items():
                state[key] = self._serialize_state(value)
            return state
        except Exception:
            return None
    
    def _serialize_state(self, value: Any, depth: int = 0) -> Any:
        """Serialize state value into JSON-friendly structure."""
        if depth > 3:
            return str(value)
        
        if isinstance(value, (int, float, bool, type(None), str)):
            return value
        if isinstance(value, (list, tuple, set)):
            return [self._serialize_state(v, depth + 1) for v in value]
        if isinstance(value, dict):
            return {str(k): self._serialize_state(v, depth + 1) for k, v in value.items()}
        if hasattr(value, "__dict__"):
            return {str(k): self._serialize_state(v, depth + 1) for k, v in value.__dict__.items()}
        return str(value)
    
    def _deep_equals(self, a: Any, b: Any) -> bool:
        """Deep equality check for complex types"""
        if a is b:
            return True
        
        # Handle numeric type coercion (int vs float) before type check
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b)) < 1e-6
        
        if type(a) != type(b):
            return False
        
        # For floats, use approximate equality
        if isinstance(a, float):
            return abs(a - b) < 1e-6
        
        if isinstance(a, dict):
            if len(a) != len(b):
                return False
            for key in a:
                if key not in b:
                    return False
                if not self._deep_equals(a[key], b[key]):
                    return False
            return True
        
        if isinstance(a, (list, tuple)):
            if len(a) != len(b):
                return False
            return all(self._deep_equals(x, y) for x, y in zip(a, b))
        
        if isinstance(a, set):
            if len(a) != len(b):
                return False
            return a == b
        
        return a == b
    
    def _separate_self_overrides(self, inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        call_arguments: Dict[str, Any] = {}
        state_overrides: Dict[str, Any] = {}
        for key, value in inputs.items():
            if isinstance(key, str) and key.startswith("self."):
                attr = key.split(".", 1)[1]
                state_overrides[attr] = value
            else:
                call_arguments[key] = value
        return call_arguments, state_overrides
    
    def _dedupe_preserve_order(self, items: Iterable[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered
    
    def _strip_future_imports_from_code(self, code: str) -> Tuple[str, List[str]]:
        future_lines: List[str] = []
        remaining_lines: List[str] = []
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("from __future__ import"):
                future_lines.append(stripped)
            else:
                remaining_lines.append(line)
        return "\n".join(remaining_lines), future_lines
    
    def _format_future_and_standard_imports(
        self,
        future_imports: List[str],
        standard_imports: List[str]
    ) -> str:
        future_block = ""
        standard_block = ""
        deduped_future = self._dedupe_preserve_order(future_imports)
        deduped_standard = self._dedupe_preserve_order(standard_imports)
        if deduped_future:
            future_block = "\n".join(deduped_future) + "\n"
        if deduped_standard:
            standard_block = "\n".join(deduped_standard) + "\n"
        if future_block and standard_block:
            return future_block + "\n" + standard_block + "\n"
        return future_block + standard_block + ("\n" if future_block or standard_block else "")
    
    def _create_temp_module_file(self, code: str, imports: List[str], prefix: str,
                                 specification: Optional[Dict[str, Any]] = None,
                                 file_path: Optional[str] = None,
                                 is_regenerated: bool = False) -> str:
        """Create a temporary module file that can be used for coverage analysis"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', prefix=f"{prefix}_", delete=False)
        import_lines: List[str] = []
        
        for imp in imports or []:
            if not imp:
                continue
            line = imp.strip()
            if line and line not in import_lines:
                import_lines.append(line)
        
        future_imports: List[str] = []
        standard_imports: List[str] = []
        for line in import_lines:
            if line.startswith("from __future__ import"):
                future_imports.append(line)
            else:
                standard_imports.append(line)
        
        module_sections: List[str] = []
        cleaned_code, snippet_future = self._strip_future_imports_from_code(textwrap.dedent(code))
        future_imports.extend(snippet_future)
        
        if specification and (specification.get('class_context') or specification.get('class_name')):
            class_name = specification.get('class_name') or specification.get('class_context', {}).get('class_name')
            if class_name and file_path:
                class_source = self._load_class_source(file_path, class_name)
            else:
                class_source = None
            
            if class_source:
                class_source_clean, class_future = self._strip_future_imports_from_code(class_source)
                future_imports.extend(class_future)
                if is_regenerated:
                    injected = self._inject_method_into_class(
                        class_source_clean,
                        specification.get('function_name'),
                        cleaned_code
                    )
                    class_source_clean = injected if injected else class_source_clean
                
                module_sections.append(class_source_clean.rstrip())
                
                instance_helper_template = textwrap.dedent("""
_last_instance = None
_instance_overrides = {}

def _heuristic_value(annotation, name):
    simple = str(annotation).lower() if annotation else ""
    if 'int' in simple:
        return 0
    if 'float' in simple:
        return 0.0
    if 'bool' in simple:
        return False
    if 'list' in simple:
        return []
    if 'dict' in simple:
        return {}
    if 'set' in simple:
        return set()
    if 'tuple' in simple:
        return tuple()
    if 'str' in simple:
        return ""
    if 'sequence' in simple:
        return []
    if name.lower().endswith('count'):
        return 0
    if name.lower().endswith('name'):
        return ""
    return None

def _build_constructor_kwargs(overrides):
    import inspect
    kwargs = {}
    try:
        signature = inspect.signature(__CLASS_NAME__)
    except Exception:
        signature = None
    if signature:
        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue
            if overrides and param_name in overrides:
                kwargs[param_name] = overrides[param_name]
                continue
            if param.default is not inspect._empty:
                kwargs[param_name] = param.default
            else:
                candidate = _heuristic_value(param.annotation, param_name)
                if candidate is not None:
                    kwargs[param_name] = candidate
        return kwargs
    return {}

def _instantiate(overrides=None):
    overrides = overrides or {}
    kwargs = _build_constructor_kwargs(overrides)
    try:
        instance = __CLASS_NAME__(**kwargs)
    except Exception:
        instance = __CLASS_NAME__.__new__(__CLASS_NAME__)
        if hasattr(instance, '__init__'):
            try:
                instance.__init__(**kwargs)
            except Exception:
                pass
    for attr, value in overrides.items():
        setattr(instance, attr, value)
    return instance

def _reset_instance(overrides=None):
    global _last_instance, _instance_overrides
    _instance_overrides = overrides or {}
    _last_instance = _instantiate(_instance_overrides)
    return _last_instance

def _get_last_instance():
    return _last_instance

def _apply_state_overrides(patch):
    global _last_instance, _instance_overrides
    if patch:
        _instance_overrides.update(patch)
    if _last_instance is None:
        _last_instance = _instantiate(_instance_overrides)
    else:
        for attr, value in (_instance_overrides or {}).items():
            setattr(_last_instance, attr, value)
    return _last_instance
""").strip()
                instance_helper = instance_helper_template.replace("__CLASS_NAME__", class_name)
                module_sections.append(instance_helper)
                
                method_name = specification.get('function_name')
                if method_name:
                    wrapper_template = textwrap.dedent("""
def __METHOD_NAME__(*args, **kwargs):
    global _last_instance
    if _last_instance is None:
        _last_instance = _instantiate(_instance_overrides)
    return _last_instance.__METHOD_NAME__(*args, **kwargs)
""").strip()
                    wrapper = wrapper_template.replace("__METHOD_NAME__", method_name)
                    module_sections.append(wrapper)
                    
                    preparer = textwrap.dedent("""
def _prepare_instance():
    _instance_overrides.clear()
    return _reset_instance(_instance_overrides)
""").strip()
                    module_sections.append(preparer)
            else:
                module_sections.append(cleaned_code.strip())
        else:
            module_sections.append(cleaned_code.strip())
        
        module_sections = [section for section in module_sections if section]
        module_body = "\n\n".join(module_sections) + "\n"
        header = self._format_future_and_standard_imports(future_imports, standard_imports)
        module_content = header + module_body
        
        temp_file.write(module_content)
        temp_file.flush()
        temp_file.close()
        
        return temp_file.name
    
    def _cleanup_temp_module(self, module_path: str) -> None:
        """Remove temporary module file if it exists"""
        if module_path and os.path.exists(module_path):
            try:
                os.remove(module_path)
            except OSError:
                pass
    
    def _load_class_source(self, file_path: str, class_name: str) -> Optional[str]:
        """Load the source code for a specific class from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
        except (OSError, IOError):
            return None
        
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return ast.get_source_segment(source, node)
        
        return None
    
    def _inject_method_into_class(self, class_source: str, method_name: Optional[str], new_method_code: str) -> Optional[str]:
        """Inject regenerated method code into the class source"""
        if not method_name:
            return None
        
        try:
            class_tree = ast.parse(class_source)
        except SyntaxError:
            return None
        
        class_node = None
        for node in class_tree.body:
            if isinstance(node, ast.ClassDef):
                class_node = node
                break
        
        if class_node is None:
            return None
        
        dedented_method = textwrap.dedent(new_method_code).strip()
        if not dedented_method.startswith("def"):
            return None
        
        indented_method = textwrap.indent(dedented_method, "    ")
        
        body_segments = []
        for elem in class_node.body:
            segment = ast.get_source_segment(class_source, elem)
            if isinstance(elem, ast.FunctionDef) and elem.name == method_name:
                body_segments.append(indented_method)
            elif segment is not None:
                body_segments.append(segment)
            else:
                body_segments.append(textwrap.indent("pass", "    "))
        
        class_header = class_source.splitlines()[0]
        reconstructed = [class_header]
        reconstructed.extend(body_segments)
        return "\n".join(reconstructed) + "\n"
    
    def _calculate_branch_coverage(self, coverage_tracker: Coverage, module_path: str, original_code: str) -> Dict[str, Any]:
        """Calculate branch coverage percentage and identify uncovered branches"""
        try:
            # Ensure coverage data is saved before accessing
            try:
                coverage_tracker.save()
            except Exception:
                pass
            
            normalized_path = os.path.realpath(module_path)
            alt_path = os.path.abspath(module_path)
            candidate_paths = [normalized_path, alt_path, module_path]
            
            data = coverage_tracker.get_data()
            
            missing_lines: List[int] = []
            missing_branches: List[Dict[str, Any]] = []
            coverage_ratio = 1.0
            executed_lines: Set[int] = set()
            
            branch_counts = {}
            measured_files: List[str] = []
            if hasattr(data, "measured_files"):
                try:
                    measured_files = list(data.measured_files())
                except Exception:
                    measured_files = []
            
            # Match measured files to our module path (handle path variations like /private/var vs /var)
            module_basename = os.path.basename(module_path)
            for measured in measured_files:
                if os.path.basename(measured) == module_basename:
                    # Add all variations of the path
                    if measured not in candidate_paths:
                        candidate_paths.append(measured)
                    if os.path.realpath(measured) not in candidate_paths:
                        candidate_paths.append(os.path.realpath(measured))
                    if os.path.abspath(measured) not in candidate_paths:
                        candidate_paths.append(os.path.abspath(measured))
            
            if not measured_files:
                print(f"        [coverage] No measured files for {module_basename}")
            else:
                matching = [m for m in measured_files if os.path.basename(m) == module_basename]
                if matching:
                    print(f"        [coverage] Found {len(matching)} measured file(s) matching {module_basename}")
                else:
                    print(f"        [coverage] Measured files exist but none match {module_basename}")
            for candidate in candidate_paths:
                if not os.path.exists(candidate):
                    print(f"        [coverage] Candidate missing on disk: {candidate}")
                try:
                    candidate_lines = data.lines(candidate) or ()
                    if candidate_lines:
                        executed_lines.update(candidate_lines)
                except Exception:
                    continue
            
            if data.has_arcs():
                # Use measured files first (most reliable)
                files_to_check = measured_files if measured_files else candidate_paths
                
                for file_path in files_to_check:
                    try:
                        arcs = data.arcs(file_path)
                        if arcs:
                            # Filter out entry/exit arcs (negative line numbers)
                            branch_arcs = [(f, t) for f, t in arcs if f > 0 and t > 0]
                            
                            if branch_arcs:
                                # Get all possible branches from AST analysis
                                import ast
                                try:
                                    tree = ast.parse(original_code)
                                    possible_branches = set()
                                    
                                    for node in ast.walk(tree):
                                        if isinstance(node, ast.If):
                                            if_line = node.lineno
                                            if node.body:
                                                then_line = node.body[0].lineno
                                                possible_branches.add((if_line, then_line))
                                            if node.orelse:
                                                else_line = node.orelse[0].lineno
                                                possible_branches.add((if_line, else_line))
                                            else:
                                                # No else clause - implicit else branch
                                                possible_branches.add((if_line, if_line + 1))
                                    
                                    # Create map of executed branches from arcs
                                    executed_branch_set = set(branch_arcs)
                                    
                                    # Match executed arcs to possible branches
                                    executed_branches = 0
                                    for branch in possible_branches:
                                        if branch in executed_branch_set:
                                            executed_branches += 1
                                        else:
                                            # Check if any arc from the same source line matches
                                            if_line = branch[0]
                                            matching_arcs = [a for a in executed_branch_set if a[0] == if_line]
                                            if matching_arcs:
                                                executed_branches += 1
                                            else:
                                                missing_branches.append({
                                                    'line': branch[0],
                                                    'target': branch[1],
                                                    'detail': f"Branch from line {branch[0]} to {branch[1]} not executed"
                                                })
                                    
                                    total_branches = len(possible_branches)
                                    
                                    if total_branches > 0:
                                        coverage_ratio = executed_branches / total_branches
                                        # Create branch_counts dict for compatibility
                                        lines_with_branches = set(b[0] for b in possible_branches)
                                        branch_counts = {}
                                        for line in lines_with_branches:
                                            line_branches = [b for b in possible_branches if b[0] == line]
                                            executed_line_branches = len([b for b in line_branches if b in executed_branch_set or any(a[0] == line for a in executed_branch_set)])
                                            branch_counts[line] = (executed_line_branches, len(line_branches))
                                        break
                                except Exception as e:
                                    print(f"        [coverage] AST analysis error: {e}")
                                    pass
                            
                            # Fallback: estimate from arcs if AST fails
                            if not branch_counts and branch_arcs:
                                # Count unique source lines with branches
                                source_lines = set(arc[0] for arc in branch_arcs)
                                # Estimate: each source line typically has 2 branches (if/else)
                                estimated_total = len(source_lines) * 2
                                executed_count = len(branch_arcs)
                                if estimated_total > 0:
                                    coverage_ratio = min(executed_count / estimated_total, 1.0)
                                    branch_counts = {line: (1, 2) for line in source_lines}
                                break
                    except Exception as e:
                        continue
                
                print(f"        [coverage] branch counts for {os.path.basename(module_path)}: {branch_counts}")
            
            if branch_counts:
                # Recalculate if not already done above
                if coverage_ratio == 1.0:
                    total_branches = 0
                    executed_branches = 0
                    
                    for line_no, (executed, total) in branch_counts.items():
                        total_branches += total
                        executed_branches += executed
                    
                    if total_branches > 0:
                        coverage_ratio = executed_branches / total_branches
            
            # Fallback to analysis2 for missing line information
            analysis_success = False
            for candidate in candidate_paths:
                try:
                    _, _, _, missing, missing_branch_text = coverage_tracker.analysis2(candidate)
                except Exception:
                    continue
                analysis_success = True
                if not missing_lines:
                    missing_lines = missing
                if not missing_branch_text:
                    break
                if not missing_branches:
                    for entry in missing_branch_text:
                        line_str, _, target = entry.partition("->")
                        try:
                            line_no = int(line_str.split()[0])
                        except Exception:
                            line_no = None
                        missing_branches.append({
                            'line': line_no,
                            'detail': entry
                        })
                    break
            
            executable_lines, docstring_lines = self._extract_executable_line_info(original_code)
            
            if not analysis_success and not branch_counts:
                # No coverage data found - check if we have executed lines
                if executed_lines and executable_lines:
                    # We have executed lines but no branch data - calculate line coverage
                    covered_lines = len([line for line in executable_lines if line in executed_lines])
                    line_cov = covered_lines / len(executable_lines) if executable_lines else 1.0
                    coverage_ratio = max(coverage_ratio, line_cov)
                elif not executed_lines:
                    # No executed lines either - likely very simple function with no branches
                    # Default to 100% coverage for simple functions
                    coverage_ratio = 1.0
            
            coverage_ratio = min(max(coverage_ratio, 0.0), 1.0)
            if executable_lines:
                covered_line_count = len([line for line in executable_lines if line in executed_lines])
                if covered_line_count or executed_lines:
                    line_coverage_ratio = covered_line_count / len(executable_lines) if executable_lines else 1.0
                    if branch_counts:
                        # Use the stricter metric when branches exist
                        coverage_ratio = min(coverage_ratio, line_coverage_ratio)
                    else:
                        coverage_ratio = line_coverage_ratio
            
            if executable_lines:
                missing_lines = [line for line in missing_lines if line in executable_lines]
            
            filtered_missing_branches = []
            for entry in missing_branches:
                line = entry.get('line')
                if line is None:
                    continue
                if executable_lines and line not in executable_lines:
                    continue
                filtered_missing_branches.append(entry)
            
            code_lines = original_code.splitlines()
            control_only_keywords = {'else:', 'else'}
            
            def _is_control_marker(line_no: Optional[int]) -> bool:
                if line_no is None:
                    return False
                if line_no <= 0 or line_no > len(code_lines):
                    return False
                stripped = code_lines[line_no - 1].strip()
                if not stripped:
                    return True
                normalized = stripped.replace(" ", "")
                if normalized in control_only_keywords:
                    return True
                return False
            
            missing_lines = [line for line in missing_lines if not _is_control_marker(line)]
            filtered_missing_branches = [
                entry for entry in filtered_missing_branches if not _is_control_marker(entry.get('line'))
            ]
            
            # Ensure coverage_ratio is always set (default to 1.0 if no branches found)
            if coverage_ratio == 1.0 and not branch_counts and not executed_lines:
                # No coverage data available - default to indicating coverage is unknown
                # For very simple functions with no branches, assume 100% coverage
                coverage_ratio = 1.0 if not executable_lines or len(executable_lines) <= 2 else 0.0
            
            coverage_complete = coverage_ratio >= 0.9999
            if executable_lines:
                if filtered_missing_branches or missing_lines:
                    coverage_complete = False
                else:
                    uncovered_lines = sorted(line for line in executable_lines if line not in executed_lines)
                    missing_lines = uncovered_lines
                    if uncovered_lines:
                        coverage_complete = False
            else:
                # No executable lines - likely simple function, assume complete coverage
                coverage_complete = True
                missing_lines = []
                filtered_missing_branches = []
            
                return {
                'coverage': coverage_ratio,
                'coverage_complete': coverage_complete,
                'missing_branches': filtered_missing_branches,
                'missing_lines': missing_lines
            }
        except Exception:
                return {
                'coverage': 0.0,
                'coverage_complete': False,
                'missing_branches': [],
                'missing_lines': []
            }
    
    def _extract_executable_line_info(self, code: str) -> Tuple[Set[int], Set[int]]:
        """Identify executable lines and docstring lines within the provided code snippet"""
        executable_lines: Set[int] = set()
        docstring_lines: Set[int] = set()
        
        cleaned_code = textwrap.dedent(code)
        
        try:
            tree = ast.parse(cleaned_code)
        except SyntaxError:
            return executable_lines, docstring_lines
        except ValueError:
            return executable_lines, docstring_lines
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = getattr(node, 'body', [])
                if body:
                    first_stmt = body[0]
                    if (
                        isinstance(first_stmt, ast.Expr)
                        and isinstance(getattr(first_stmt, 'value', None), ast.Constant)
                        and isinstance(first_stmt.value.value, str)
                    ):
                        lineno = getattr(first_stmt, 'lineno', None)
                        if lineno is not None:
                            end_lineno = getattr(first_stmt, 'end_lineno', lineno)
                            docstring_lines.update(range(lineno, end_lineno + 1))
        
        for node in ast.walk(tree):
            if isinstance(node, ast.stmt):
                if (
                    isinstance(node, ast.Expr)
                    and isinstance(getattr(node, 'value', None), ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    continue
                lineno = getattr(node, 'lineno', None)
                if lineno is None:
                    continue
                end_lineno = getattr(node, 'end_lineno', lineno)
                executable_lines.update(range(lineno, end_lineno + 1))
        
        if docstring_lines:
            executable_lines.difference_update(docstring_lines)
        
        code_lines = cleaned_code.splitlines()
        control_markers = {'else:', 'else', 'finally:', 'finally', 'try:', 'try'}
        prunable = []
        for line_no in executable_lines:
            if line_no <= 0 or line_no > len(code_lines):
                continue
            stripped = code_lines[line_no - 1].strip()
            if not stripped:
                prunable.append(line_no)
                continue
            normalized = stripped.replace(" ", "")
            if normalized in control_markers:
                prunable.append(line_no)
        for line_no in prunable:
            executable_lines.discard(line_no)
        
        return executable_lines, docstring_lines


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
            
            print(f"      Analyzing {specifications[func_id]['function_name']}...")
            
            try:
                # Get original code from context (NOT from specification)
                original_code = context.get('original_code', {}).get(func_id, '')
                if not original_code:
                    # Fallback: try to get from func_info if available
                    func_info = context.get('function_info', {}).get(func_id, {})
                    original_code = func_info.get('source_code', '')
                
                if func_id not in regenerated_code:
                    print(f"        WARNING: No regenerated code for {func_id}")
                    similarity_results[func_id] = {
                        'original_code': original_code,
                        'regenerated_code': None,
                        'similarity_metrics': {
                            'textual_similarity': 0.0,
                            'structural_similarity': 0.0,
                            'behavioral_similarity': 0.0,
                            'behavioral_test_similarity': 0.0,
                            'branch_coverage': 0.0,
                            'primary_similarity': 0.0,
                            'regeneration_failed': True,
                            'error': 'Code regeneration returned empty result'
                        },
                        'test_based_validation': False
                    }
                    continue
                
                regen_code = regenerated_code[func_id].get('code', '')
                if not regen_code:
                    print(f"        WARNING: Empty regenerated code for {func_id}")
                    similarity_results[func_id] = {
                        'original_code': original_code,
                        'regenerated_code': None,
                        'similarity_metrics': {
                            'textual_similarity': 0.0,
                            'structural_similarity': 0.0,
                            'behavioral_similarity': 0.0,
                            'behavioral_test_similarity': 0.0,
                            'branch_coverage': 0.0,
                            'primary_similarity': 0.0,
                            'regeneration_failed': True,
                            'error': 'Code regeneration returned empty result'
                        },
                        'test_based_validation': False
                    }
                    continue
                
                similarity_metrics = self.semantic_analyzer.calculate_semantic_similarity(
                    original_code, regen_code
                )
                
                # Check for semantic equivalence
                from agents.semantic_equivalence import SemanticEquivalenceDetector
                equiv_detector = SemanticEquivalenceDetector()
                textual_sim = _coerce_to_float(similarity_metrics.get('textual_similarity', 0.0))
                struct_sim = _coerce_to_float(similarity_metrics.get('structural_similarity', 0.0))
                behav_sim = _coerce_to_float(similarity_metrics.get('behavioral_similarity', 0.0))
                adj_struct, adj_behav = equiv_detector.adjust_similarity_for_equivalence(
                    original_code, regen_code, struct_sim, behav_sim
                )
                similarity_metrics['textual_similarity'] = textual_sim
                similarity_metrics['structural_similarity'] = adj_struct
                similarity_metrics['behavioral_similarity'] = adj_behav
                
                if func_id in test_results:
                    test_data = test_results[func_id]
                    behavioral_test_similarity = self._calculate_behavioral_test_similarity(test_data)
                    similarity_metrics['behavioral_test_similarity'] = behavioral_test_similarity
                    similarity_metrics['branch_coverage'] = test_data.get('branch_coverage', 0.0)
                    similarity_metrics['coverage_complete'] = test_data.get('coverage_complete', False)
                    similarity_metrics['missing_branches'] = test_data.get('missing_branches', [])
                    similarity_metrics['missing_lines'] = test_data.get('missing_lines', [])
                else:
                    similarity_metrics['behavioral_test_similarity'] = 0.0
                    similarity_metrics['branch_coverage'] = 0.0
                    similarity_metrics['coverage_complete'] = False
                    similarity_metrics['missing_branches'] = []
                    similarity_metrics['missing_lines'] = []
                
                textual_similarity = _coerce_to_float(similarity_metrics.get('textual_similarity', 0.0))
                structural_similarity = _coerce_to_float(similarity_metrics.get('structural_similarity', 0.0))
                behavioral_similarity = _coerce_to_float(similarity_metrics.get('behavioral_similarity', 0.0))
                behavioral_test_similarity = _coerce_to_float(
                    similarity_metrics.get("behavioral_test_similarity", 0.0)
                )

                tt = (
                    int(test_results[func_id].get("total_tests", 0))
                    if func_id in test_results
                    else 0
                )
                primary_similarity = compute_primary_similarity_metrics(
                    structural_similarity,
                    behavioral_test_similarity,
                    tt,
                    config=self.config,
                )
                similarity_metrics["primary_similarity"] = primary_similarity
                similarity_metrics["textual_similarity"] = textual_similarity
                similarity_metrics['structural_similarity'] = structural_similarity
                similarity_metrics['behavioral_similarity'] = behavioral_similarity
                similarity_metrics['behavioral_test_similarity'] = behavioral_test_similarity
                
                similarity_results[func_id] = {
                    'original_code': original_code,
                    'regenerated_code': regen_code,
                    'similarity_metrics': similarity_metrics,
                    'test_based_validation': func_id in test_results
                }
                
                print(f"        Textual similarity: {textual_similarity:.1%}")
                print(f"        Structural similarity: {structural_similarity:.1%}")
                print(f"        Behavioral similarity: {behavioral_similarity:.1%}")
                if func_id in test_results:
                    print(f"        Behavioral test similarity: {behavioral_test_similarity:.1%} "
                          f"(branch coverage {_coerce_to_float(similarity_metrics['branch_coverage']):.1%})")
                    if not similarity_metrics.get('coverage_complete', False):
                        missing_lines = similarity_metrics.get('missing_lines', [])
                        if missing_lines:
                            print(f"          Missing coverage at lines: {missing_lines}")
                else:
                    print("        Behavioral test similarity: Not available (tests missing)")
                
            except Exception as e:
                print(f"        ERROR: {e}")
                continue
        
        context['similarity_results'] = similarity_results
        
        # Store original_code in context for results analysis
        if 'original_code' not in context:
            context['original_code'] = {}
        for func_id, result in similarity_results.items():
            if 'original_code' in result:
                context['original_code'][func_id] = result['original_code']
        
        similarities = [r['similarity_metrics'].get('primary_similarity', 0.0) for r in similarity_results.values()]
        if 'similarity_history' not in context:
            context['similarity_history'] = []
        context['similarity_history'].extend(similarities)
        
        print(f"    Analyzed {len(similarity_results)} functions")
        
        return context
    
    def _calculate_behavioral_test_similarity(self, test_results: Dict[str, Any]) -> float:
        """Calculate behavioral test similarity based on test pass rates"""
        # Try multiple ways to get total_tests
        total_tests = test_results.get('total_tests', 0)
        if not total_tests:
            # Try to calculate from passed/failed counts
            original_passed = test_results.get('original_passed', 0)
            original_failed = test_results.get('original_failed', 0)
            regenerated_passed = test_results.get('regenerated_passed', 0)
            regenerated_failed = test_results.get('regenerated_failed', 0)
            total_tests = original_passed + original_failed
            if not total_tests:
                total_tests = regenerated_passed + regenerated_failed
        
        total_tests = int(total_tests) if total_tests else 0
        
        if total_tests <= 0:
            return 0.0
        
        # Calculate pass rate based on behavioral match (both original and regenerated must pass)
        original_passed = test_results.get('original_passed', 0)
        regenerated_passed = test_results.get('regenerated_passed', 0)
        
        # Behavioral similarity = how often both original AND regenerated pass the same test
        failures = test_results.get('failures', [])
        if not isinstance(failures, list):
            return 0.0
        behavioral_matches = total_tests - len(failures)
        pass_rate = behavioral_matches / total_tests if total_tests > 0 else 0.0
        
        # Focus on test pass rate - this is the most important metric
        # Branch coverage tracking with dynamic imports is unreliable, so we rely on test results
        return pass_rate
    

class FailureDrivenSpecRefinementNode(BaseNode):
    """
    Failure-driven spec refinement: analyzes diff between original and regenerated
    code to infer missing abstract specification content. Adds natural language
    updates (no code snippets) and triggers regeneration.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from utils.call_llm import call_llm
        from agents.failure_driven_refinement import FailureDrivenRefinementEngine
        self.call_llm = partial(call_llm, config=self.config, system=SYSTEM_FAILURE_DRIVEN_REFINEMENT)
        self.refinement_engine = FailureDrivenRefinementEngine(self.call_llm)
        self.semantic_analyzer = SemanticSimilarityAnalyzer()
        self.max_attempts = config.get('failure_driven_max_attempts', 3)
        self.min_improvement = config.get('min_improvement_for_early_exit', 0.02)
        self.target_similarity = config.get('target_similarity', 1.0)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        For each function below threshold, analyze diff, infer abstract spec
        updates, merge into spec, and trigger regeneration. Up to max_attempts
        per function, with early exit if improvement is negligible.
        """
        target = context.get('target_similarity', self.target_similarity)
        specifications = context.get('specifications', {})
        similarity_results = context.get('similarity_results', {})
        original_code_dict = context.get('original_code', {})
        regenerated_code = context.get('regenerated_code', {})

        failure_driven_results = context.get('failure_driven_results', {})

        for func_id, spec_data in specifications.items():
            if not spec_data.get('success', False):
                continue
            if func_id not in similarity_results:
                continue

            metrics = similarity_results[func_id].get('similarity_metrics', {})
            primary = metrics.get('primary_similarity', 0.0)
            if primary >= target:
                continue

            original_code = original_code_dict.get(func_id, '')
            regen_code = regenerated_code.get(func_id, {}).get('code', '')
            if not original_code or not regen_code:
                continue

            print(f"      Failure-driven refinement for {spec_data['function_name']} "
                  f"(current: {primary:.1%}, target: {target:.1%})...")

            prev_similarity = primary
            attempts = 0
            phase_succeeded = False

            while attempts < self.max_attempts:
                attempts += 1
                spec = spec_data['specification']
                updates = self.refinement_engine.analyze_diff_and_infer_spec_updates(
                    original_code, regen_code,
                    spec_data['function_name'],
                    spec,
                    metrics
                )
                self.refinement_engine.merge_updates_into_spec(spec, updates)

                # Regenerate and re-analyze
                from nodes import CodeRegenerationNode, TestGenerationNode, TestExecutionNode
                regen_node = CodeRegenerationNode(self.config)
                test_gen_node = TestGenerationNode(self.config)
                test_exec_node = TestExecutionNode(self.config)

                if func_id in context.get('regenerated_code', {}):
                    del context['regenerated_code'][func_id]
                context = regen_node.execute(context)

                if func_id not in context.get('regenerated_code', {}):
                    break

                # Run tests and similarity for this function
                context = test_gen_node.execute(context)
                context = test_exec_node.execute(context)
                new_regen = context['regenerated_code'][func_id].get('code', '')
                if new_regen:
                    new_metrics = self.semantic_analyzer.calculate_semantic_similarity(
                        original_code, new_regen
                    )
                    test_data = context.get('test_results', {}).get(func_id, {}) or {}
                    behav_sim = self._calc_behavioral_test_sim(test_data) if test_data else 0.0
                    new_metrics['behavioral_test_similarity'] = behav_sim
                    struct = new_metrics.get('structural_similarity', 0.0)
                    tt = (
                        test_data.get("total_tests", 0)
                        or (
                            test_data.get("original_passed", 0)
                            + test_data.get("original_failed", 0)
                        )
                    )
                    new_primary = compute_primary_similarity_metrics(
                        struct,
                        behav_sim,
                        int(tt),
                        config=self.config,
                    )
                    similarity_results[func_id]['similarity_metrics'] = new_metrics
                    similarity_results[func_id]['similarity_metrics']['primary_similarity'] = new_primary
                    regen_code = new_regen

                    improvement = new_primary - prev_similarity
                    if new_primary >= target:
                        phase_succeeded = True
                        print(f"        Attempt {attempts}: {new_primary:.1%} - target reached")
                        break
                    if improvement < self.min_improvement:
                        print(f"        Attempt {attempts}: {new_primary:.1%} - minimal improvement, stopping")
                        break
                    prev_similarity = new_primary
                    print(f"        Attempt {attempts}: {new_primary:.1%} (improved by {improvement:.1%})")
                else:
                    break

            failure_driven_results[func_id] = {
                'attempts': attempts,
                'initial_similarity': primary,
                'final_similarity': prev_similarity,
                'phase_succeeded': phase_succeeded,
                'phase': 'failure_driven'
            }
            print(f"        Completed: {attempts} attempts, final {prev_similarity:.1%}")

        context['failure_driven_results'] = failure_driven_results
        context['similarity_results'] = similarity_results
        return context

    def _calc_behavioral_test_sim(self, test_data: Dict[str, Any]) -> float:
        total = test_data.get('total_tests', 0) or (
            test_data.get('original_passed', 0) + test_data.get('original_failed', 0)
        )
        if total <= 0:
            return 0.0
        failures = test_data.get('failures', [])
        matches = total - len(failures) if isinstance(failures, list) else 0
        return matches / total


class _HybridPiece:
    """Wrapper for hybrid code pieces (diff-driven or tier-based) with unified .code and .tier interface."""
    def __init__(self, code: str, tier: str = 'diff_driven', description: str = ''):
        self.code = code
        self.tier = tier
        self.description = description


class HybridSpecsNode(BaseNode):
    """
    Diagnostic incremental hybrid (v2): rank gap-derived snippets, regen, backtrack on
    low marginal gain, escalate minimal lines to full statement blocks when needed.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from utils.code_diff_analyzer import CodeDiffAnalyzer
        from utils.hybrid_gap_planner import HybridGapPlanner
        from utils.smart_code_extractor import SmartCodeExtractor

        self.diff_analyzer = CodeDiffAnalyzer
        self.gap_planner = HybridGapPlanner()
        self.smart_extractor = SmartCodeExtractor()
        self.max_iterations = config.get("hybrid_max_iterations", 12)
        self.similarity_threshold = config.get("hybrid_similarity_threshold", 0.999)
        self.min_improvement = float(config.get("hybrid_min_improvement_per_step", 0.015))
        self.max_regens_per_func = int(config.get("hybrid_max_regens_per_func", 5))
        self.allow_full_fallback = bool(config.get("hybrid_allow_full_code_fallback", False))

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("    Running diagnostic hybrid specs generation...")

        specifications = context.get("specifications", {})
        regenerated_code = context.get("regenerated_code", {})
        similarity_results = context.get("similarity_results", {})
        original_code_dict = context.get("original_code", {})
        hybrid_specs = context.get("hybrid_specs", {})

        for func_id, spec_data in specifications.items():
            if not spec_data.get("success", False):
                continue
            if func_id not in similarity_results:
                continue

            metrics = similarity_results[func_id].get("similarity_metrics", {})
            primary_similarity = float(metrics.get("primary_similarity", 0.0))
            if primary_similarity >= self.similarity_threshold:
                continue

            original_code = original_code_dict.get(func_id, "")
            regen_code = regenerated_code.get(func_id, {}).get("code", "")
            if not original_code or not regen_code:
                continue

            print(
                f"      Processing hybrid specs for {spec_data['function_name']} "
                f"(current similarity: {primary_similarity:.1%})..."
            )

            context, hybrid_entry = self._process_function_diagnostic_hybrid(
                context,
                func_id,
                spec_data,
                original_code,
                regen_code,
                primary_similarity,
                similarity_results,
            )
            hybrid_specs[func_id] = hybrid_entry

        context["hybrid_specs"] = hybrid_specs
        context["similarity_results"] = similarity_results
        print(f"    Processed {len(hybrid_specs)} functions for hybrid specs")
        return context

    def _process_function_diagnostic_hybrid(
        self,
        context: Dict[str, Any],
        func_id: str,
        spec_data: Dict[str, Any],
        original_code: str,
        regen_code: str,
        initial_similarity: float,
        similarity_results: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        spec = spec_data["specification"]
        current_similarity = initial_similarity
        added_pieces: List[Dict[str, Any]] = []
        rejected_ids: Set[str] = set()
        escalate_keys: Set[str] = set()
        regen_count = 0

        # NLP-first: diff descriptions only, no code graft
        plan = self.gap_planner.plan(original_code, regen_code)
        if plan.gap_summary:
            spec["hybrid_diff_summary"] = plan.gap_summary
            context, new_sim, new_regen, _ = self._hybrid_regen_and_score(
                context, func_id, original_code, persist=True
            )
            regen_count += 1
            if new_regen:
                regen_code = new_regen
            if new_sim >= self.similarity_threshold:
                print(f"        NLP-first succeeded: {new_sim:.1%} (0% code added)")
                similarity_results[func_id]["similarity_metrics"]["primary_similarity"] = new_sim
                return context, {
                    "function_name": spec_data["function_name"],
                    "initial_similarity": initial_similarity,
                    "final_similarity": new_sim,
                    "iterations": 0,
                    "regen_calls": regen_count,
                    "added_pieces": [],
                    "pieces_count": 0,
                    "method": "nlp_only",
                }
            if new_sim > current_similarity:
                current_similarity = new_sim

        already_added: List[str] = list(spec.get("hybrid_code_additions") or [])
        consecutive_failures = 0
        last_good_regen = regen_code

        while (
            current_similarity < self.similarity_threshold
            and regen_count < self.max_regens_per_func
            and consecutive_failures < 4
        ):
            test_data = context.get("test_results", {}).get(func_id, {})
            plan = self.gap_planner.plan(
                original_code,
                regen_code,
                already_added=already_added,
                rejected_ids=rejected_ids,
                escalate_stmt_keys=escalate_keys,
                test_data=test_data,
            )
            if not plan.candidates:
                break

            cand = plan.candidates[0]
            sim_before = current_similarity
            HybridGapPlanner.append_addition(spec, cand.code)
            already_added.append(cand.code)

            print(
                f"        Hybrid step {regen_count + 1}: +[{cand.category}] "
                f"score={cand.score:.0f} — {cand.description[:55]}"
            )

            context, new_sim, new_regen, new_metrics = self._hybrid_regen_and_score(
                context, func_id, original_code, persist=False
            )
            regen_count += 1
            delta = new_sim - sim_before

            piece_record = {
                "code": cand.code,
                "iteration": regen_count,
                "method": f"diagnostic_{cand.source}",
                "category": cand.category,
                "candidate_id": cand.candidate_id,
                "score": cand.score,
                "delta_similarity": delta,
                "escalation_level": cand.escalation_level,
            }

            if delta >= self.min_improvement or new_sim >= self.similarity_threshold:
                current_similarity = new_sim
                added_pieces.append(piece_record)
                consecutive_failures = 0
                if new_metrics:
                    similarity_results[func_id]["similarity_metrics"] = new_metrics
                if new_regen:
                    regen_code = new_regen
                    last_good_regen = new_regen
                    context.setdefault("regenerated_code", {})[func_id] = {
                        "code": new_regen,
                        "function_name": spec_data["function_name"],
                    }
                print(f"          -> {new_sim:.1%} (Δ{delta:+.1%})")
                if new_sim >= self.similarity_threshold:
                    break
                continue

            # Backtrack: snippet did not help enough (or regressed)
            print(
                f"          -> backtrack {new_sim:.1%} (Δ{delta:+.1%} < {self.min_improvement:.1%})"
            )
            HybridGapPlanner.remove_last_addition(spec, cand.code)
            if already_added and already_added[-1] == cand.code:
                already_added.pop()
            rejected_ids.add(cand.candidate_id)
            consecutive_failures += 1
            if cand.stmt_key and cand.escalation_level == 0:
                escalate_keys.add(cand.stmt_key)
            regen_code = last_good_regen
            if last_good_regen and func_id in context.get("regenerated_code", {}):
                context["regenerated_code"][func_id]["code"] = last_good_regen

        # Optional legacy full-code fallback (off by default)
        if (
            self.allow_full_fallback
            and current_similarity < self.similarity_threshold
            and regen_count < self.max_regens_per_func
        ):
            spec["hybrid_code_additions"] = [original_code]
            spec["hybrid_use_exact_code"] = True
            added_pieces.append(
                {
                    "code": original_code,
                    "iteration": regen_count + 1,
                    "method": "last_resort_full_code",
                }
            )
            print("        Last resort: full original code in spec (fallback enabled)")
            context, new_sim, new_regen, _ = self._hybrid_regen_and_score(
                context, func_id, original_code, persist=True
            )
            regen_count += 1
            current_similarity = new_sim
            if new_regen:
                regen_code = new_regen

        similarity_results[func_id]["similarity_metrics"]["primary_similarity"] = current_similarity

        entry = {
            "function_name": spec_data["function_name"],
            "initial_similarity": initial_similarity,
            "final_similarity": current_similarity,
            "iterations": len(added_pieces),
            "regen_calls": regen_count,
            "added_pieces": added_pieces,
            "pieces_count": len(added_pieces),
            "method": "diagnostic_incremental",
            "rejected_candidates": len(rejected_ids),
            "escalated_stmt_keys": sorted(escalate_keys),
        }
        print(
            f"        Completed: {len(added_pieces)} kept piece(s), "
            f"{regen_count} regen call(s), final {current_similarity:.1%}"
        )
        return context, entry

    def _hybrid_regen_and_score(
        self,
        context: Dict[str, Any],
        func_id: str,
        original_code: str,
        *,
        persist: bool = True,
    ) -> Tuple[Dict[str, Any], float, str, Dict[str, Any]]:
        """Regenerate one function, re-run tests, return updated primary similarity."""
        from agents.advanced_analyzer import SemanticSimilarityAnalyzer
        from nodes import CodeRegenerationNode, TestExecutionNode, TestGenerationNode

        regen_node = CodeRegenerationNode(self.config)
        test_gen = TestGenerationNode(self.config)
        test_exec = TestExecutionNode(self.config)
        analyzer = SemanticSimilarityAnalyzer()

        if func_id in context.get("regenerated_code", {}):
            del context["regenerated_code"][func_id]

        context = regen_node.execute(context)
        new_regen = context.get("regenerated_code", {}).get(func_id, {}).get("code", "")
        if not new_regen:
            sim = float(
                context.get("similarity_results", {})
                .get(func_id, {})
                .get("similarity_metrics", {})
                .get("primary_similarity", 0.0)
            )
            return context, sim, "", {}

        context = test_gen.execute(context)
        context = test_exec.execute(context)

        new_metrics = analyzer.calculate_semantic_similarity(original_code, new_regen)
        test_data = context.get("test_results", {}).get(func_id, {})
        behav_sim = self._hybrid_behavioral_test_sim(test_data) if test_data else 0.0
        new_metrics["behavioral_test_similarity"] = behav_sim
        struct = float(new_metrics.get("structural_similarity", 0.0))
        tt = int(
            test_data.get("total_tests", 0)
            or (test_data.get("original_passed", 0) + test_data.get("original_failed", 0))
        )
        primary = compute_primary_similarity_metrics(
            struct, behav_sim, tt, config=self.config
        )
        new_metrics["primary_similarity"] = primary

        if persist:
            if func_id not in context.get("similarity_results", {}):
                context.setdefault("similarity_results", {})[func_id] = {}
            context["similarity_results"][func_id]["similarity_metrics"] = new_metrics
            context["similarity_results"][func_id]["regenerated_code"] = new_regen
        return context, primary, new_regen, new_metrics

    def _hybrid_behavioral_test_sim(self, test_data: Dict[str, Any]) -> float:
        total = test_data.get("total_tests", 0) or (
            test_data.get("original_passed", 0) + test_data.get("original_failed", 0)
        )
        if total <= 0:
            return 0.0
        failures = test_data.get("failures", [])
        if not isinstance(failures, list):
            return 0.0
        return (total - len(failures)) / total


class FeedbackLoopNode(BaseNode):
    """First feedback loop: Modifies prompt based on similarity gaps and natural language diff descriptions"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.smart_prompt_engine = SmartPromptEngine()
        from utils.code_diff_analyzer import CodeDiffAnalyzer
        self.diff_analyzer = CodeDiffAnalyzer
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process first feedback loop: modify prompts with metric gaps + natural language diff descriptions"""
        print("    Processing feedback loop (prompt modification)...")
        
        similarity_results = context.get('similarity_results', {})
        target_similarity = context['target_similarity']
        original_code_dict = context.get('original_code', {})
        regenerated_code = context.get('regenerated_code', {})
        test_results = context.get('test_results', {})
        
        if 'feedback_data' not in context:
            context['feedback_data'] = {}
        
        improved_count = 0
        
        for func_id, result in similarity_results.items():
            metrics = result.get('similarity_metrics', {})
            primary_similarity = metrics.get('primary_similarity', 0.0)
            
            if primary_similarity < target_similarity:
                print(f"      Analyzing gaps for {func_id}...")
                
                try:
                    gaps = self._analyze_similarity_gaps(result)
                    
                    # Add natural language diff descriptions for refinement loop
                    original_code = original_code_dict.get(func_id, '')
                    regen_code = regenerated_code.get(func_id, {}).get('code', '')
                    diff_descriptions = []
                    if original_code and regen_code:
                        diff_descriptions = self.diff_analyzer.get_diff_natural_language_descriptions(
                            original_code, regen_code, max_descriptions=12
                        )
                        if diff_descriptions:
                            gaps.append("CODE DIFF (original vs regenerated - fix these in the spec):")
                            gaps.extend(diff_descriptions)

                    tr = test_results.get(func_id)
                    if tr and not tr.get('behavioral_match', True):
                        rf_node = RuntimeFeedbackLoopNode(self.config)
                        test_detail = rf_node._summarize_test_failures(tr)
                        if test_detail:
                            gaps.append(
                                "CONCRETE TEST OUTPUT MISMATCHES (align spec with original behavior): "
                                + test_detail
                            )
                    
                    context['feedback_data'][func_id] = {
                        'gaps': gaps,
                        'diff_descriptions': diff_descriptions,
                        'metrics': result['similarity_metrics'],
                        'iteration': context.get('current_iteration', 1)
                    }
                    
                    improved_count += 1
                    print(f"        Feedback prepared ({len(gaps)} gap items)")
                
                except Exception as e:
                    print(f"        ERROR: {e}")
                    continue
        
        print(f"    Prepared feedback for {improved_count} functions")
        
        return context
    
    def _analyze_similarity_gaps(self, result: Dict[str, Any]) -> List[str]:
        """Analyze gaps in similarity (metric-based)"""
        gaps = []
        metrics = result['similarity_metrics']
        
        if _coerce_to_float(metrics.get('structural_similarity', 0)) < 0.8:
            gaps.append("Structural differences: Code organization and AST structure differ")
        
        if _coerce_to_float(metrics.get('behavioral_similarity', 0)) < 0.8:
            gaps.append("Behavioral differences: Function behavior patterns differ")
        
        behavioral_test_sim = _coerce_to_float(metrics.get('behavioral_test_similarity', 0))
        if behavioral_test_sim < 0.8:
            if metrics.get('coverage_complete', False):
                gaps.append("Test-based validation: Functions produce different outputs for same inputs")
            else:
                branch_cov = _coerce_to_float(metrics.get('branch_coverage', 0))
                gaps.append(f"Test-based validation: Low behavioral test similarity ({behavioral_test_sim:.1%}) with {branch_cov:.1%} branch coverage")

        struct_sim = _coerce_to_float(metrics.get('structural_similarity', 0))
        if behavioral_test_sim >= 0.99 and struct_sim < 0.92:
            gaps.append(
                "Structural parity lag while behavioral tests agree strongly: revise the specification so regenerated "
                "code matches original identifiers, branch structure, literals, exception classes, and call sites "
                "(AST-aligned with the reference), not merely pass the harness."
            )
        
        if _coerce_to_float(metrics.get('branch_coverage', 0)) < 0.9999:
            missing_lines = metrics.get('missing_lines', [])
            if missing_lines:
                gaps.append(f"Insufficient test coverage: Missing branches at lines {missing_lines}")
            else:
                gaps.append("Insufficient test coverage: Branch coverage below 100% for generated tests")
        
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
                        'total_failures': len(results.get('failures', []))
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
    
    def _summarize_test_failures(self, test_results: Dict[str, Any]) -> str:
        """Summarize test failures into actionable feedback"""
        failures = test_results.get('failures', [])
        if not failures:
            return ""
        
        summaries = []
        for failure in failures[:5]:
            test = failure.get('test', {})
            test_name = test.get('test_name', 'Unknown test')
            inputs = test.get('inputs', {})
            original_out = failure.get('original_output')
            regen_out = failure.get('regenerated_output')
            original_exc = failure.get('original_exception')
            regen_exc = failure.get('regenerated_exception')
            
            if original_exc or regen_exc:
                if original_exc and not regen_exc:
                    summary = f"Test '{test_name}' with inputs {inputs}: Original raises {original_exc}, regenerated returns {regen_out} (should raise exception)"
                elif not original_exc and regen_exc:
                    summary = f"Test '{test_name}' with inputs {inputs}: Original returns {original_out}, regenerated raises {regen_exc} (should not raise)"
                else:
                    summary = f"Test '{test_name}' with inputs {inputs}: Original raises {original_exc}, regenerated raises {regen_exc} (exception mismatch)"
            else:
                summary = f"Test '{test_name}' with inputs {inputs}: Expected {original_out}, got {regen_out}"
            
            summaries.append(summary)
        
        if len(failures) > 5:
            summaries.append(f"... and {len(failures) - 5} more failures")
        
        return "; ".join(summaries)


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
            if result.get('similarity_metrics', {}).get('primary_similarity', 0.0) >= target_similarity
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
        elif current_iteration >= 2:
            recent_similarities = context.get('similarity_history', [])[-total_functions:]
            if len(recent_similarities) >= total_functions:
                prev_similarities = context.get('similarity_history', [])[-2*total_functions:-total_functions]
                if len(prev_similarities) >= total_functions:
                    avg_improvement = np.mean(recent_similarities) - np.mean(prev_similarities)
                    # Check for functions with 0% similarity that need more iterations
                    zero_similarity_count = sum(1 for s in recent_similarities if s < 0.01)
                    if zero_similarity_count > 0 and current_iteration < max_iterations:
                        # Don't converge if we have failed regenerations
                        converged = False
                        reason = f"{zero_similarity_count} functions still have 0% similarity, continuing..."
                    elif avg_improvement < 0.003 and current_iteration >= 3:
                        converged = True
                        reason = f"No significant improvement (avg change: {avg_improvement:.3%}) after {current_iteration} iterations"
                    elif avg_improvement < 0.015 and current_iteration >= 4:
                        converged = True
                        reason = f"Minimal improvement (avg change: {avg_improvement:.3%}) after {current_iteration} iterations"
        
        context['convergence_achieved'] = converged
        context['convergence_reason'] = reason
        context['convergence_rate'] = convergence_rate
        
        if converged:
            print(f"    Convergence achieved: {reason}")
        else:
            print(f"    Continuing iteration (convergence rate: {convergence_rate:.1%})")
        
        return context

