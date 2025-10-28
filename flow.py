"""
Enhanced PocketFlow Orchestration Engine V2
Includes test generation, execution, and dual feedback loops
"""

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
    FeedbackLoopNode,
    RuntimeFeedbackLoopNode,
    ConvergenceCheckerNode
)


class EnhancedPocketFlowOrchestrator:
    """Enhanced orchestrator with test generation and dual feedback loops"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workflow_graph = nx.DiGraph()
        self.results = {}
        self._setup_workflow()
    
    def _setup_workflow(self):
        """Setup the workflow graph"""
        self.workflow_graph.add_node("code_analyzer", node=CodeAnalyzerNode(self.config))
        self.workflow_graph.add_node("spec_generator", node=SpecificationGeneratorNode(self.config))
        self.workflow_graph.add_node("code_regeneration", node=CodeRegenerationNode(self.config))
        self.workflow_graph.add_node("test_generation", node=TestGenerationNode(self.config))
        self.workflow_graph.add_node("test_execution", node=TestExecutionNode(self.config))
        self.workflow_graph.add_node("similarity_analyzer", node=SimilarityAnalyzerNode(self.config))
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
        
        context = {
            'project_path': project_path,
            'target_similarity': target_similarity,
            'max_iterations': self.config.get('max_iterations', 10),
            'current_iteration': 0,
            'specifications': {},
            'regenerated_code': {},
            'generated_tests': {},
            'test_results': {},
            'similarity_results': {},
            'similarity_history': [],
            'feedback_data': {},
            'runtime_feedback': {},
            'convergence_achieved': False
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
                except Exception as e:
                    print(f"    ERROR: convergence_checker failed: {e}")
                    raise
            
            if context.get('convergence_achieved', False):
                print(f"Convergence achieved after {iteration} iterations")
                break
        
        if iteration >= max_iterations:
            print(f"WARNING: Maximum iterations ({max_iterations}) reached")
        
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
        
        target_achieved_count = sum(
            1 for func_id, result in context.get('similarity_results', {}).items()
            if result.get('overall_similarity', 0) >= context['target_similarity']
        )
        
        total_functions = len(context['specifications'])
        success_rate = target_achieved_count / total_functions if total_functions > 0 else 0.0
        
        test_stats = self._calculate_test_statistics(context)
        
        analysis = {
            'total_functions': total_functions,
            'successful_functions': len([s for s in context['specifications'].values() if s.get('success', False)]),
            'failed_functions': len([s for s in context['specifications'].values() if not s.get('success', False)]),
            'average_similarity': avg_similarity,
            'success_rate': success_rate,
            'target_achieved_count': target_achieved_count,
            'iterations_completed': context['current_iteration'],
            'convergence_achieved': context.get('convergence_achieved', False),
            'similarity_distribution': self._analyze_similarity_distribution(similarities),
            'test_statistics': test_stats,
            'function_results': self._compile_function_results(context)
        }
        
        return {
            'success': True,
            'analysis': analysis,
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
                'behavioral_matches': 0,
                'behavioral_mismatches': 0
            }
        
        total_tests = sum(r['total_tests'] for r in test_results.values())
        behavioral_matches = sum(1 for r in test_results.values() if r.get('behavioral_match', False))
        
        return {
            'tests_generated': len(context.get('generated_tests', {})),
            'tests_executed': len(test_results),
            'total_test_cases': total_tests,
            'behavioral_matches': behavioral_matches,
            'behavioral_mismatches': len(test_results) - behavioral_matches,
            'behavioral_match_rate': behavioral_matches / len(test_results) if test_results else 0.0
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
    
    def _compile_function_results(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compile detailed results for each function"""
        results = {}
        
        for func_id, spec in context.get('specifications', {}).items():
            if not spec.get('success', False):
                results[func_id] = {
                    'success': False,
                    'error': spec.get('error', 'Unknown error')
                }
                continue
            
            similarity_data = context.get('similarity_results', {}).get(func_id, {})
            test_data = context.get('test_results', {}).get(func_id, {})
            
            results[func_id] = {
                'success': True,
                'function_name': spec['function_name'],
                'file_path': spec['file_path'],
                'final_similarity': similarity_data.get('overall_similarity', 0.0),
                'similarity_metrics': similarity_data.get('similarity_metrics', {}),
                'tests_executed': test_data.get('total_tests', 0) if test_data else 0,
                'behavioral_match': test_data.get('behavioral_match', False) if test_data else False,
                'test_pass_rate': (test_data['regenerated_passed'] / test_data['total_tests'] 
                                  if test_data and test_data['total_tests'] > 0 else 0.0)
            }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "enhanced_output"):
        """Save enhanced results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "enhanced_pocketflow_results.json"), 'w') as f:
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


def create_enhanced_pocketflow_orchestrator(config: Dict[str, Any]) -> EnhancedPocketFlowOrchestrator:
    """Factory function to create Enhanced PocketFlow orchestrator"""
    return EnhancedPocketFlowOrchestrator(config)

