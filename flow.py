"""
Test generation, execution, and dual feedback loops
"""

import copy
import os
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import networkx as nx
from nodes import (
    CodeAnalyzerNode,
    SpecificationGeneratorNode,
    CodeRegenerationNode,
    TestGenerationNode,
    TestExecutionNode,
    SimilarityAnalyzerNode,
    FailureDrivenSpecRefinementNode,
    HybridSpecsNode,
    FeedbackLoopNode,
    RuntimeFeedbackLoopNode,
    ConvergenceCheckerNode
)


class SpecificationOrchestrator:
    """Orchestrator for specification generation with test execution and feedback loops"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workflow_graph = nx.DiGraph()
        self.results = {}
        self._setup_workflow()

    def _build_run_config(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Snapshot of hyperparameters for reproducibility (paper alignment, ablations)."""
        c = self.config
        return {
            'target_similarity': context.get('target_similarity', c.get('target_similarity', 0.99)),
            'max_iterations': context.get('max_iterations', c.get('max_iterations', 10)),
            'enable_program_slicing': c.get('enable_program_slicing', True),
            'enable_failure_driven_refinement': c.get('enable_failure_driven_refinement', True),
            'enable_hybrid_specs': c.get('enable_hybrid_specs', True),
            'hybrid_similarity_threshold': c.get('hybrid_similarity_threshold', 0.99),
            'hybrid_min_improvement_per_step': c.get('hybrid_min_improvement_per_step', 0.015),
            'hybrid_max_regens_per_func': c.get('hybrid_max_regens_per_func', 5),
            'hybrid_allow_full_code_fallback': c.get('hybrid_allow_full_code_fallback', False),
            'failure_driven_max_attempts': c.get('failure_driven_max_attempts', 3),
            'hybrid_max_iterations': c.get('hybrid_max_iterations', 12),
            'min_improvement_for_early_exit': c.get('min_improvement_for_early_exit', 0.02),
            'model': c.get('model', ''),
            'only_qualified_names': c.get('only_qualified_names'),
            'ablation_profile': self._infer_ablation_profile(c),
            'enable_context_bundle': c.get('enable_context_bundle', True),
            'context_budget_chars': c.get('context_budget_chars'),
            'context_k_hop': c.get('context_k_hop'),
            'context_scope_parent_levels': c.get('context_scope_parent_levels'),
            'context_enable_rag': c.get('context_enable_rag', False),
            'context_spec_prompt_inject_chars': c.get('context_spec_prompt_inject_chars'),
            'context_regen_bundle_chars': c.get('context_regen_bundle_chars'),
            'enable_llm_generated_tests': c.get('enable_llm_generated_tests', True),
            'min_behavioral_cases': c.get('min_behavioral_cases'),
            'max_llm_test_generation_rounds_per_func': c.get(
                'max_llm_test_generation_rounds_per_func'
            ),
            'harness_mode': c.get('harness_mode', 'auto'),
            'pytest_harness_timeout_sec': c.get('pytest_harness_timeout_sec', 300),
            'context_test_prompt_bundle_bytes': c.get('context_test_prompt_bundle_bytes'),
            'regeneration_spec_json_char_budget': c.get('regeneration_spec_json_char_budget'),
            'trusted_behavioral_oracle_blend': c.get('trusted_behavioral_oracle_blend', True),
            'trusted_behavioral_agreement_floor': c.get('trusted_behavioral_agreement_floor'),
            'behavioral_oracle_blend_weight': c.get('behavioral_oracle_blend_weight'),
        }

    @staticmethod
    def _infer_ablation_profile(config: Dict[str, Any]) -> str:
        """Human-readable label for the active configuration."""
        s = config.get('enable_program_slicing', True)
        fd = config.get('enable_failure_driven_refinement', True)
        h = config.get('enable_hybrid_specs', True)
        if s and fd and h:
            return 'full_pipeline'
        if not s and not fd and not h:
            return 'minimal_stages'
        if not s and fd and h:
            return 'baseline_monolithic_no_slicing'
        if s and not fd and not h:
            return 'baseline_no_stage2_no_stage3'
        if s and fd and not h:
            return 'no_hybrid_stage3'
        if s and not fd and h:
            return 'no_failure_driven_stage2'
        return 'custom'
    
    def _setup_workflow(self):
        """Setup the workflow graph"""
        self.workflow_graph.add_node("code_analyzer", node=CodeAnalyzerNode(self.config))
        self.workflow_graph.add_node("spec_generator", node=SpecificationGeneratorNode(self.config))
        self.workflow_graph.add_node("code_regeneration", node=CodeRegenerationNode(self.config))
        self.workflow_graph.add_node("test_generation", node=TestGenerationNode(self.config))
        self.workflow_graph.add_node("test_execution", node=TestExecutionNode(self.config))
        self.workflow_graph.add_node("similarity_analyzer", node=SimilarityAnalyzerNode(self.config))
        self.workflow_graph.add_node("failure_driven_refinement", node=FailureDrivenSpecRefinementNode(self.config))
        self.workflow_graph.add_node("hybrid_specs", node=HybridSpecsNode(self.config))
        self.workflow_graph.add_node("feedback_loop", node=FeedbackLoopNode(self.config))
        self.workflow_graph.add_node("runtime_feedback_loop", node=RuntimeFeedbackLoopNode(self.config))
        self.workflow_graph.add_node("convergence_checker", node=ConvergenceCheckerNode(self.config))
        
        self.workflow_graph.add_edge("code_analyzer", "spec_generator")
        self.workflow_graph.add_edge("spec_generator", "code_regeneration")
        self.workflow_graph.add_edge("code_regeneration", "test_generation")
        self.workflow_graph.add_edge("test_generation", "test_execution")
        self.workflow_graph.add_edge("test_execution", "similarity_analyzer")
        self.workflow_graph.add_edge("similarity_analyzer", "feedback_loop")
        self.workflow_graph.add_edge("feedback_loop", "runtime_feedback_loop")
        self.workflow_graph.add_edge("runtime_feedback_loop", "convergence_checker")
    
    def process_project(self, project_path: str, target_similarity: float = 0.95) -> Dict[str, Any]:
        """Process a Python project through the workflow"""
        print(f"Starting specification generation for: {project_path}")
        print(f"   Features: Test generation, dual feedback loops, behavioral validation")
        
        base_max_iterations = self.config.get('max_iterations', 10)
        context = {
            'project_path': project_path,
            'target_similarity': target_similarity,
            'max_iterations': base_max_iterations,
            'base_max_iterations': base_max_iterations,
            'current_iteration': 0,
            'specifications': {},
            'regenerated_code': {},
            'generated_tests': {},
            'test_results': {},
            'similarity_results': {},
            'similarity_history': [],
            'feedback_data': {},
            'runtime_feedback': {},
            'convergence_achieved': False,
            'function_complexities': {}
        }
        
        try:
            context = self._execute_workflow(context)
            final_report = self._generate_final_report(context)
            return final_report
            
        except Exception as e:
            print(f"ERROR in processing: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'context': context
            }
    
    def _execute_workflow(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow with dual feedback loops"""
        iteration = 0
        max_iterations = context['max_iterations']
        
        code_analyzer_executed = False
        
        while iteration < max_iterations and not context.get('convergence_achieved', False):
            iteration += 1
            context['current_iteration'] = iteration
            
            print(f"\nIteration {iteration}")
            print("=" * 70)
            
            if not code_analyzer_executed:
                node = self.workflow_graph.nodes["code_analyzer"]['node']
                print(f"  Executing code_analyzer...")
                try:
                    context = node.execute(context)
                    print(f"    code_analyzer completed")
                    code_analyzer_executed = True
                except Exception as e:
                    print(f"    ERROR: code_analyzer failed: {e}")
                    raise
            
            core_nodes = [
                "spec_generator",
                "code_regeneration",
                "test_generation",
                "test_execution",
                "similarity_analyzer",
                "feedback_loop",
                "runtime_feedback_loop"
            ]
            
            for node_name in core_nodes:
                if context.get('convergence_achieved', False):
                    break
                
                node = self.workflow_graph.nodes[node_name]['node']
                print(f"  Executing {node_name}...")
                
                try:
                    context = node.execute(context)
                    print(f"    {node_name} completed")
                except Exception as e:
                    print(f"    ERROR: {node_name} failed: {e}")
                    raise
            
            if not context.get('convergence_achieved', False):
                node = self.workflow_graph.nodes["convergence_checker"]['node']
                print(f"  Executing convergence_checker...")
                
                try:
                    context = node.execute(context)
                    print(f"    convergence_checker completed")
                    
                    # Adapt max_iterations based on complexity after first iteration
                    if iteration == 1 and context.get('function_complexities'):
                        from agents.divide_conquer import DeltaImprovementAlgorithm
                        delta_improver = DeltaImprovementAlgorithm()
                        max_complexity = max(context['function_complexities'].values()) if context['function_complexities'] else 1
                        adaptive_max = delta_improver.calculate_adaptive_iterations({
                            'complexity': max_complexity,
                            'cyclomatic_complexity': max_complexity * 1.2,
                            'num_paths': max_complexity // 2,
                            'branching_factor': max_complexity // 3
                        }, base_iterations=context.get('base_max_iterations', 3))
                        if adaptive_max > context['max_iterations']:
                            context['max_iterations'] = min(adaptive_max, 10)
                            print(f"    Adjusted max_iterations to {context['max_iterations']} based on complexity")
                except Exception as e:
                    print(f"    ERROR: convergence_checker failed: {e}")
                    raise
            
            # Capture similarity after first run (no feedback loops, no hybrid)
            if iteration == 1 and 'similarity_after_first_run' not in context:
                sr = context.get('similarity_results', {})
                context['similarity_after_first_run'] = {
                    fid: {'similarity_metrics': copy.deepcopy(res.get('similarity_metrics', {}))}
                    for fid, res in sr.items()
                }
            
            if context.get('convergence_achieved', False):
                print(f"Convergence achieved after {iteration} iterations")
                break

        if iteration >= max_iterations:
            print(f"WARNING: Maximum iterations ({max_iterations}) reached")

        # Stage 2: Failure-driven refinement for functions below threshold
        target = context.get('target_similarity', 0.95)
        below_threshold = [
            fid for fid, res in context.get('similarity_results', {}).items()
            if res.get('similarity_metrics', {}).get('primary_similarity', 0.0) < target
        ]
        if below_threshold and self.config.get('enable_failure_driven_refinement', True):
            print(f"\nStage 2: Failure-driven refinement ({len(below_threshold)} functions below threshold)")
            print("=" * 70)
            context['target_similarity'] = target
            context['min_improvement_for_early_exit'] = self.config.get('min_improvement_for_early_exit', 0.02)
            try:
                fd_node = self.workflow_graph.nodes["failure_driven_refinement"]['node']
                context = fd_node.execute(context)
            except Exception as e:
                print(f"  ERROR: failure_driven_refinement failed: {e}")
                import traceback
                traceback.print_exc()

        # Capture similarity after feedback loops (before hybrid)
        sr = context.get('similarity_results', {})
        context['similarity_after_feedback'] = {
            fid: {'similarity_metrics': copy.deepcopy(res.get('similarity_metrics', {}))}
            for fid, res in sr.items()
        }

        # Stage 3: Hybrid specs for functions still below threshold
        below_threshold = [
            fid for fid, res in context.get('similarity_results', {}).items()
            if res.get('similarity_metrics', {}).get('primary_similarity', 0.0) < target
        ]
        if below_threshold and self.config.get('enable_hybrid_specs', True):
            for fid in below_threshold:
                res = context['similarity_results'].get(fid, {})
                m = res.get('similarity_metrics', {})
                if 'best_without_hybrid' not in context:
                    context['best_without_hybrid'] = {}
                context['best_without_hybrid'][fid] = {
                    'textual': m.get('textual_similarity', 0.0),
                    'structural': m.get('structural_similarity', 0.0),
                    'behavioral_test': m.get('behavioral_test_similarity', 0.0),
                }
            print(f"\nStage 3: Hybrid specs ({len(below_threshold)} functions still below threshold)")
            print("=" * 70)
            try:
                hybrid_node = self.workflow_graph.nodes["hybrid_specs"]['node']
                context = hybrid_node.execute(context)
            except Exception as e:
                print(f"  ERROR: hybrid_specs failed: {e}")
                import traceback
                traceback.print_exc()

        return context
    
    def _generate_final_report(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final report of the enhanced specification generation process"""
        if not context.get('specifications'):
            return {
                'success': False,
                'error': 'No specifications generated',
                'context': context
            }
        
        similarities = context.get('similarity_history', [])
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        textual_values = []
        structural_values = []
        behavioral_values = []
        behavioral_test_values = []
        branch_coverage_values = []
        primary_values = []
        
        for result in context.get('similarity_results', {}).values():
            metrics = result.get('similarity_metrics', {})
            textual_values.append(metrics.get('textual_similarity', 0.0))
            structural_values.append(metrics.get('structural_similarity', 0.0))
            behavioral_values.append(metrics.get('behavioral_similarity', 0.0))
            behavioral_test_values.append(metrics.get('behavioral_test_similarity', 0.0))
            branch_coverage_values.append(metrics.get('branch_coverage', 0.0))
            primary_values.append(metrics.get('primary_similarity', 0.0))
        
        def _average(values: List[float]) -> float:
            return sum(values) / len(values) if values else 0.0
        
        target_achieved_count = sum(
            1 for func_id, result in context.get('similarity_results', {}).items()
            if result.get('similarity_metrics', {}).get('primary_similarity', 0.0) >= context['target_similarity']
        )
        
        total_functions = len(context['specifications'])
        success_rate = target_achieved_count / total_functions if total_functions > 0 else 0.0
        
        test_stats = self._calculate_test_statistics(context)
        
        # Behavioral test similarity: project-level = (# tests with same results) / (# total tests)
        # Not per-function average (which penalizes functions with 0 tests)
        test_results = context.get('test_results', {})
        total_tests_project = sum(r.get('total_tests', 0) for r in test_results.values())
        if total_tests_project > 0:
            total_matches_project = sum(
                r.get('total_tests', 0) - len(r.get('failures', []))
                for r in test_results.values()
            )
            behavioral_test_avg = total_matches_project / total_tests_project
        else:
            behavioral_test_avg = _average(behavioral_test_values)
        
        # ``primary_similarity`` is defined per-function in ``SimilarityAnalyzerNode`` /
        # hybrid updates (often structural-only until enough harness tests exist).
        textual_avg = _average(textual_values)
        ast_avg = _average(structural_values)
        behavioral_avg = _average(behavioral_values)
        # Per-function ``primary_similarity`` already applies min-test-case blending rules in
        # ``SimilarityAnalyzerNode`` — keep the headline metric consistent with that.
        primary_avg = _average(primary_values)
        phase_tracking = self._compile_phase_tracking(context)
        paper_data = self._compile_paper_data(context)
        context['paper_data'] = paper_data  # Needed for _compile_loop_stats
        loop_stats = self._compile_loop_stats(context)
        context['phase_tracking'] = phase_tracking
        context['loop_stats'] = loop_stats
        run_config = self._build_run_config(context)
        analysis = {
            'total_functions': total_functions,
            'successful_functions': len([s for s in context['specifications'].values() if s.get('success', False)]),
            'failed_functions': len([s for s in context['specifications'].values() if not s.get('success', False)]),
            'phase_tracking': phase_tracking,
            'loop_stats': loop_stats,
            'average_primary_similarity': primary_avg,  # Calculated from overall averages for consistency
            'average_textual_similarity': textual_avg,
            'average_structural_similarity': ast_avg,
            'average_behavioral_similarity': behavioral_avg,
            'average_behavioral_test_similarity': behavioral_test_avg,
            'average_branch_coverage': _average(branch_coverage_values),
            'success_rate': success_rate,
            'target_achieved_count': target_achieved_count,
            'iterations_completed': context['current_iteration'],
            'convergence_achieved': context.get('convergence_achieved', False),
            'similarity_distribution': self._analyze_similarity_distribution(similarities),
            'test_statistics': test_stats,
            'paper_data': paper_data,
            'function_results': self._compile_function_results(context, phase_tracking),
            'run_config': run_config,
        }
        
        return {
            'success': True,
            'analysis': analysis,
            'run_config': run_config,
            'context': context,
            'timestamp': time.time()
        }
    
    def _calculate_test_statistics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate test execution statistics"""
        test_results = context.get('test_results', {})
        
        if not test_results:
            return {
                'tests_generated': 0,
                'tests_executed': 0,
                'total_test_cases': 0,
                'test_cases_matching': 0,
                'test_case_agreement_rate': 0.0,
                'behavioral_matches': 0,
                'behavioral_mismatches': 0,
                'behavioral_match_rate': 0.0,
                'full_branch_coverage': 0,
            }
        
        total_tests = sum(r['total_tests'] for r in test_results.values())
        matching_cases = sum(
            r.get('total_tests', 0) - len(r.get('failures') or [])
            for r in test_results.values()
        )
        behavioral_matches = sum(1 for r in test_results.values() if r.get('behavioral_match', False))
        full_branch_coverage = sum(1 for r in test_results.values() if r.get('coverage_complete', False))
        
        return {
            'tests_generated': len(context.get('generated_tests', {})),
            'tests_executed': len(test_results),
            'total_test_cases': total_tests,
            'test_cases_matching': matching_cases,
            'test_case_agreement_rate': matching_cases / total_tests if total_tests > 0 else 0.0,
            'behavioral_matches': behavioral_matches,
            'behavioral_mismatches': len(test_results) - behavioral_matches,
            'behavioral_match_rate': behavioral_matches / len(test_results) if test_results else 0.0,
            'full_branch_coverage': full_branch_coverage
        }
    
    def _analyze_similarity_distribution(self, similarities: List[float]) -> Dict[str, int]:
        """Analyze distribution of similarity scores"""
        distribution = {
            'excellent (≥95%)': 0,
            'very_good (≥90%)': 0,
            'good (≥85%)': 0,
            'fair (≥70%)': 0,
            'poor (<70%)': 0
        }
        
        for similarity in similarities:
            if similarity >= 0.95:
                distribution['excellent (≥95%)'] += 1
            elif similarity >= 0.90:
                distribution['very_good (≥90%)'] += 1
            elif similarity >= 0.85:
                distribution['good (≥85%)'] += 1
            elif similarity >= 0.70:
                distribution['fair (≥70%)'] += 1
            else:
                distribution['poor (<70%)'] += 1
        
        return distribution
    
    def _project_level_behavioral(self, test_results: Dict[str, Any]) -> float:
        """Project-level behavioral similarity: (# tests with same results) / (# total tests)."""
        if not test_results:
            return 0.0
        total_tests = sum(r.get('total_tests', 0) for r in test_results.values())
        if total_tests <= 0:
            return 0.0
        total_matches = sum(
            r.get('total_tests', 0) - len(r.get('failures', []) or [])
            for r in test_results.values()
        )
        return total_matches / total_tests

    def _compile_loop_stats(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compile similarity at each stage: first run, after feedback, hybrid % code."""
        def _avg_metrics(data: Dict[str, Any]) -> Dict[str, float]:
            if not data:
                return {'textual': 0.0, 'structural': 0.0, 'behavioral_test': 0.0, 'primary': 0.0}
            vals = {'textual': [], 'structural': [], 'behavioral_test': [], 'primary': []}
            for res in data.values():
                m = res.get('similarity_metrics', {})
                vals['textual'].append(m.get('textual_similarity', 0.0))
                vals['structural'].append(m.get('structural_similarity', 0.0))
                vals['behavioral_test'].append(m.get('behavioral_test_similarity', 0.0))
                vals['primary'].append(m.get('primary_similarity', 0.0))
            n = len(vals['textual'])
            return {
                'textual': sum(vals['textual']) / n if n else 0.0,
                'structural': sum(vals['structural']) / n if n else 0.0,
                'behavioral_test': sum(vals['behavioral_test']) / n if n else 0.0,
                'primary': sum(vals['primary']) / n if n else 0.0,
            }

        test_results = context.get('test_results', {})
        project_behav = self._project_level_behavioral(test_results)
        total_tests = sum(r.get('total_tests', 0) for r in test_results.values())
        use_project_behav = total_tests > 0

        first_run = context.get('similarity_after_first_run', {})
        after_feedback = context.get('similarity_after_feedback', {})
        paper_data = context.get('paper_data', {})
        hybrid_results = context.get('hybrid_specs', {})
        
        hybrid_code_percents = [
            pd.get('hybrid_code_percent', 0.0)
            for pd in (paper_data.get('per_function', {}) or {}).values()
            if pd.get('hybrid_code_percent', 0.0) > 0
        ]
        
        s1 = _avg_metrics(first_run)
        s2 = _avg_metrics(after_feedback)
        behav1 = project_behav if use_project_behav else s1.get('behavioral_test', 0.0)
        behav2 = project_behav if use_project_behav else s2.get('behavioral_test', 0.0)

        return {
            'stage1_first_run': {
                'description': 'Similarity after first run (no feedback loops, no hybrid)',
                'functions': len(first_run),
                'avg_textual': s1.get('textual', 0.0),
                'avg_structural': s1.get('structural', 0.0),
                'avg_behavioral_test': behav1,
                'avg_primary': s1.get('primary', 0.0),
            },
            'stage2_after_feedback': {
                'description': 'Similarity after feedback loops (failure-driven)',
                'functions': len(after_feedback),
                'avg_textual': s2.get('textual', 0.0),
                'avg_structural': s2.get('structural', 0.0),
                'avg_behavioral_test': behav2,
                'avg_primary': s2.get('primary', 0.0),
            },
            'stage3_hybrid': {
                'description': '% of original code needed to achieve 100% (hybrid only)',
                'functions_using_hybrid': len(hybrid_code_percents),
                'avg_code_percent': sum(hybrid_code_percents) / len(hybrid_code_percents) if hybrid_code_percents else 0.0,
                'min_code_percent': min(hybrid_code_percents) if hybrid_code_percents else 0.0,
                'max_code_percent': max(hybrid_code_percents) if hybrid_code_percents else 0.0,
            },
        }

    def _compile_paper_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compile data for paper: best without hybrid, refinement loops, hybrid % code, hybrid loops."""
        similarity_results = context.get('similarity_results', {})
        fd_results = context.get('failure_driven_results', {})
        hybrid_results = context.get('hybrid_specs', {})
        specifications = context.get('specifications', {})
        original_code = context.get('original_code', {})

        best_without_hybrid_ctx = context.get('best_without_hybrid', {})
        per_function = {}
        for func_id in similarity_results:
            metrics = similarity_results[func_id].get('similarity_metrics', {})
            fd_res = fd_results.get(func_id, {})
            hybrid_res = hybrid_results.get(func_id, {})
            spec_data = specifications.get(func_id, {})
            spec = spec_data.get('specification', {}) if spec_data else {}
            orig_code = original_code.get(func_id, '')

            if func_id in best_without_hybrid_ctx:
                best_without_hybrid = best_without_hybrid_ctx[func_id]
            else:
                best_without_hybrid = {
                    'textual': metrics.get('textual_similarity', 0.0),
                    'structural': metrics.get('structural_similarity', 0.0),
                    'behavioral_test': metrics.get('behavioral_test_similarity', 0.0),
                }
            refinement_loops_needed = fd_res.get('attempts', 0)
            hybrid_loops_used = hybrid_res.get('iterations', 0)
            hybrid_code_percent = 0.0

            if hybrid_res and hybrid_res.get('iterations', 0) > 0 and orig_code:
                additions = spec.get('hybrid_code_additions', [])
                additions = [a for a in additions if str(a).strip()]
                if additions:
                    # Volume of grafted material vs full original (repeat iterations count toward total).
                    added_chars = sum(len(str(a).strip()) for a in additions)
                    orig_chars = len(orig_code.strip())
                    hybrid_code_percent = (added_chars / orig_chars * 100) if orig_chars > 0 else 0.0
                    if hybrid_code_percent == 0 and hybrid_res.get('iterations', 0) > 0:
                        hybrid_code_percent = 1.0  # Minimum: hybrid ran with non-empty bookkeeping

            first_run = context.get('similarity_after_first_run', {}).get(func_id, {})
            after_feedback = context.get('similarity_after_feedback', {}).get(func_id, {})
            m1 = first_run.get('similarity_metrics', {})
            m2 = after_feedback.get('similarity_metrics', {})

            per_function[func_id] = {
                'similarity_after_first_run': {
                    'textual': m1.get('textual_similarity', 0.0),
                    'structural': m1.get('structural_similarity', 0.0),
                    'behavioral_test': m1.get('behavioral_test_similarity', 0.0),
                    'primary': m1.get('primary_similarity', 0.0),
                },
                'similarity_after_feedback': {
                    'textual': m2.get('textual_similarity', 0.0),
                    'structural': m2.get('structural_similarity', 0.0),
                    'behavioral_test': m2.get('behavioral_test_similarity', 0.0),
                    'primary': m2.get('primary_similarity', 0.0),
                },
                'best_without_hybrid': best_without_hybrid,
                'refinement_loops_needed': refinement_loops_needed,
                'hybrid_loops_used': hybrid_loops_used,
                'hybrid_code_percent': hybrid_code_percent,
            }

        return {'per_function': per_function}

    def _compile_phase_tracking(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compile which phase succeeded for each function (normal, failure_driven, hybrid)."""
        target = context.get('target_similarity', 0.95)
        tracking = {'normal': 0, 'failure_driven': 0, 'hybrid': 0, 'none': 0}
        per_function = {}
        fd_results = context.get('failure_driven_results', {})
        hybrid_results = context.get('hybrid_specs', {})
        for func_id, res in context.get('similarity_results', {}).items():
            primary = res.get('similarity_metrics', {}).get('primary_similarity', 0.0)
            fd_res = fd_results.get(func_id, {})
            hybrid_res = hybrid_results.get(func_id, {})
            if primary < target:
                per_function[func_id] = 'none'
                tracking['none'] += 1
            elif func_id in hybrid_results and hybrid_res.get('final_similarity', 0) >= target:
                per_function[func_id] = 'hybrid'
                tracking['hybrid'] += 1
            elif func_id in fd_results and fd_res.get('phase_succeeded', False):
                per_function[func_id] = 'failure_driven'
                tracking['failure_driven'] += 1
            else:
                per_function[func_id] = 'normal'
                tracking['normal'] += 1
        return {'summary': tracking, 'per_function': per_function}

    def _compile_function_results(
        self, context: Dict[str, Any], phase_tracking: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Compile detailed results for each function"""
        results = {}
        pt = (phase_tracking or {}).get('per_function', {})
        
        for func_id, spec in context.get('specifications', {}).items():
            if not spec.get('success', False):
                results[func_id] = {
                    'success': False,
                    'error': spec.get('error', 'Unknown error')
                }
                continue
            
            similarity_data = context.get('similarity_results', {}).get(func_id, {})
            test_data = context.get('test_results', {}).get(func_id, {})
            
            phase = pt.get(func_id, 'normal')
            results[func_id] = {
                'success': True,
                'function_name': spec['function_name'],
                'file_path': spec['file_path'],
                'phase_succeeded': phase,
                'final_similarity': similarity_data.get('similarity_metrics', {}).get('primary_similarity', 0.0),
                'similarity_metrics': similarity_data.get('similarity_metrics', {}),
                'tests_executed': test_data.get('total_tests', 0) if test_data else 0,
                'behavioral_match': test_data.get('behavioral_match', False) if test_data else False,
                'test_pass_rate': (test_data['regenerated_passed'] / test_data['total_tests'] 
                                  if test_data and test_data['total_tests'] > 0 else 0.0),
                'branch_coverage': test_data.get('branch_coverage', 0.0) if test_data else 0.0,
                'coverage_complete': test_data.get('coverage_complete', False) if test_data else False,
                'missing_branches': test_data.get('missing_branches', []) if test_data else []
            }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "enhanced_output"):
        """Save enhanced results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Top-level run_config aids reproducibility without digging into analysis
        if 'run_config' not in results and results.get('analysis', {}).get('run_config'):
            results = {**results, 'run_config': results['analysis']['run_config']}
        with open(os.path.join(output_dir, "spec_results.json"), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        if 'context' in results:
            context = results['context']
            
            if 'specifications' in context:
                specs_dir = os.path.join(output_dir, "specifications")
                os.makedirs(specs_dir, exist_ok=True)
                
                for func_id, spec_data in context['specifications'].items():
                    if spec_data.get('success', False):
                        filename = func_id.replace("::", "_").replace("/", "_").replace("\\", "_") + ".json"
                        with open(os.path.join(specs_dir, filename), 'w') as f:
                            json.dump(spec_data, f, indent=2, default=str)
            
            if 'generated_tests' in context:
                tests_dir = os.path.join(output_dir, "generated_tests")
                os.makedirs(tests_dir, exist_ok=True)
                
                for func_id, test_data in context['generated_tests'].items():
                    filename = func_id.replace("::", "_").replace("/", "_").replace("\\", "_") + ".json"
                    with open(os.path.join(tests_dir, filename), 'w') as f:
                        json.dump(test_data, f, indent=2, default=str)
            
            if 'test_results' in context:
                results_dir = os.path.join(output_dir, "test_results")
                os.makedirs(results_dir, exist_ok=True)
                
                for func_id, result_data in context['test_results'].items():
                    filename = func_id.replace("::", "_").replace("/", "_").replace("\\", "_") + ".json"
                    with open(os.path.join(results_dir, filename), 'w') as f:
                        json.dump(result_data, f, indent=2, default=str)
        
        print(f"Results saved to: {output_dir}")


def create_spec_orchestrator(config: Dict[str, Any]) -> SpecificationOrchestrator:
    """Factory function to create the specification orchestrator"""
    return SpecificationOrchestrator(config)

