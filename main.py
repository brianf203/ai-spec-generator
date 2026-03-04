"""
Command-line interface for specification generation system
"""

import os
import sys
from pathlib import Path

# Load .env if it exists (before parsing args)
_env_path = Path(__file__).parent / '.env'
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, _, val = line.partition('=')
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
import json
import time
import argparse
from typing import Dict, Any, Optional

from flow import create_spec_orchestrator
from utils.call_llm import test_llm_connection


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Code Specification Generator with Test Generation and Feedback Loops",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a GitHub repository
  python enhanced_main_v2.py --repo https://github.com/username/python-project

  # Analyze a local directory
  python enhanced_main_v2.py --dir /path/to/python/project

  # Set target similarity threshold
  python enhanced_main_v2.py --dir ./test_project --target-similarity 0.95

Features:
  - Automated test generation for behavioral validation
  - Dual feedback loops: prompt modification + test failure accumulation
  - Comprehensive similarity analysis with test-based validation
  - No hardcoding or fallbacks - real testing only
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--repo", help="GitHub repository URL to analyze")
    input_group.add_argument("--dir", help="Local directory path to analyze")
    
    parser.add_argument("-n", "--name", help="Project name (derived from URL/directory if omitted)")
    parser.add_argument("-t", "--token", help="GitHub token (or set GITHUB_TOKEN environment variable)")
    parser.add_argument("-o", "--output", default="enhanced_output_v2", help="Output directory (default: ./enhanced_output_v2)")
    parser.add_argument("-i", "--include", nargs="*", default=["*.py"], help="Files to include (default: *.py)")
    parser.add_argument("-e", "--exclude", nargs="*", default=["*test*", "test_*", "tests/*", "__pycache__/*"], help="Files/dirs to exclude (e.g. *test*, test_*, tests/*)")
    parser.add_argument("-s", "--max-size", type=int, default=100000, help="Maximum file size in bytes (default: 100KB)")
    parser.add_argument("--target-similarity", type=float, default=1.0, help="Target similarity: 100%% AST + 100%% behavioral (default: 1.0)")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum iterations (default: 10)")
    parser.add_argument("--api-key", help="Gemini API key (or set GEMINI_API_KEY environment variable)")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Gemini model (default: gemini-2.0-flash)")
    parser.add_argument("--no-cache", action="store_true", help="Disable LLM response caching")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--enable-hybrid-specs", action="store_true", help="Enable hybrid specs (automatic code addition)")
    
    args = parser.parse_args()
    
    if args.repo and not args.repo.startswith(('http://', 'https://')):
        print("ERROR: Repository URL must start with http:// or https://")
        sys.exit(1)
    
    if args.dir:
        if not os.path.exists(args.dir):
            print(f"ERROR: Path not found: {args.dir}")
            sys.exit(1)
        if not os.path.isdir(args.dir):
            print(f"ERROR: Input must be a directory (folder), not a file: {args.dir}")
            print(f"   The system analyzes all Python files in a project folder.")
            sys.exit(1)
    
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("ERROR: GEMINI_API_KEY not provided")
        print("   Set environment variable: export GEMINI_API_KEY='your_key'")
        print("   Or use --api-key argument")
        sys.exit(1)
    
    os.environ["GEMINI_API_KEY"] = api_key
    
    from utils.call_llm import init_llm_from_config
    config = {
        'api_key': api_key,
        'model': args.model,
        'target_similarity': args.target_similarity,
        'max_iterations': args.max_iterations,
        'max_file_size': args.max_size,
        'include_patterns': args.include,
        'exclude_patterns': args.exclude,
        'output_dir': args.output,
        'verbose': args.verbose,
        'cache_enabled': not args.no_cache,
        'enable_hybrid_specs': True,  # Always enabled - part of three-stage pipeline
        'failure_driven_max_attempts': 3,
        'hybrid_max_iterations': 12,
        'min_improvement_for_early_exit': 0.02,
        'hybrid_similarity_threshold': 1.0,
    }
    init_llm_from_config(config)
    
    print("Testing LLM connection...")
    if not test_llm_connection():
        print("ERROR: Failed to connect to Gemini API")
        print("   Please check your API key and internet connection")
        sys.exit(1)
    
    print("Connected to Gemini API")
    
    if args.repo:
        project_path = args.repo
        project_name = args.name or extract_repo_name(args.repo)
    else:
        project_path = args.dir
        project_name = args.name or os.path.basename(os.path.abspath(args.dir))
    
    print(f"\n{'='*70}")
    print(f"Project: {project_name}")
    print(f"Target similarity: {args.target_similarity:.1%}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Features: Test generation + Dual feedback loops")
    print(f"{'='*70}")
    
    os.makedirs(args.output, exist_ok=True)
    
    print("\nInitializing specification orchestrator...")
    orchestrator = create_spec_orchestrator(config)
    
    start_time = time.time()
    
    try:
        results = orchestrator.process_project(project_path, args.target_similarity)
        end_time = time.time()
        
        orchestrator.save_results(results, args.output)
        
        if results['success']:
            analysis = results['analysis']
            print(f"\n{'='*70}")
            print(f"Specification generation completed!")
            print(f"{'='*70}")
            print(f"Total time: {end_time - start_time:.2f} seconds")
            print(f"\nFunction Statistics:")
            print(f"  Total functions: {analysis['total_functions']}")
            print(f"  Successful: {analysis['successful_functions']}")
            print(f"  Failed: {analysis['failed_functions']}")
            print(f"\nSimilarity Metrics (Textual, AST, Behavioral Test):")
            print(f"  Textual similarity:        {analysis['average_textual_similarity']:.1%}")
            print(f"  AST (structural):          {analysis['average_structural_similarity']:.1%}")
            print(f"  Behavioral test similarity:{analysis['average_behavioral_test_similarity']:.1%}")
            print(f"\nIteration Metrics:")
            print(f"  Iterations completed: {analysis['iterations_completed']}")
            print(f"  Convergence achieved: {analysis['convergence_achieved']}")

            phase_tracking = analysis.get('phase_tracking', {}).get('summary', {})
            if phase_tracking:
                print(f"\nPhase Tracking (which stage succeeded):")
                print(f"  Normal (Stage 1): {phase_tracking.get('normal', 0)}")
                print(f"  Failure-driven (Stage 2): {phase_tracking.get('failure_driven', 0)}")
                print(f"  Hybrid (Stage 3): {phase_tracking.get('hybrid', 0)}")
                print(f"  Below threshold: {phase_tracking.get('none', 0)}")

            loop_stats = analysis.get('loop_stats', {})
            if loop_stats:
                print(f"\nSimilarity by Stage:")
                s1 = loop_stats.get('stage1_first_run', {})
                if s1.get('functions', 0) > 0:
                    print(f"  Stage 1 (first run, no feedback/hybrid): textual={s1.get('avg_textual', 0):.1%} AST={s1.get('avg_structural', 0):.1%} behavioral_test={s1.get('avg_behavioral_test', 0):.1%}")
                s2 = loop_stats.get('stage2_after_feedback', {})
                if s2.get('functions', 0) > 0:
                    print(f"  Stage 2 (after feedback loops):         textual={s2.get('avg_textual', 0):.1%} AST={s2.get('avg_structural', 0):.1%} behavioral_test={s2.get('avg_behavioral_test', 0):.1%}")
                s3 = loop_stats.get('stage3_hybrid', {})
                if s3.get('functions_using_hybrid', 0) > 0:
                    print(f"  Stage 3 (hybrid code % to reach 100%):   avg={s3.get('avg_code_percent', 0):.1f}% min={s3.get('min_code_percent', 0):.1f}% max={s3.get('max_code_percent', 0):.1f}%")

            # Hybrid code % summary (only for functions that used hybrid)
            paper_data = analysis.get('paper_data', {}).get('per_function', {})
            hybrid_pcts = [pd.get('hybrid_code_percent', 0) for fid, pd in paper_data.items()
                          if pd.get('hybrid_loops_used', 0) > 0]
            if hybrid_pcts:
                print(f"\nHybrid Specs (code % used to achieve 100%):")
                print(f"  Functions using hybrid: {len(hybrid_pcts)}")
                print(f"  Avg code %: {sum(hybrid_pcts)/len(hybrid_pcts):.1f}%")
                print(f"  Min/Max: {min(hybrid_pcts):.1f}% / {max(hybrid_pcts):.1f}%")

            if paper_data:
                print(f"\nPer-function details (first 5):")
                for func_id, pd in list(paper_data.items())[:5]:
                    fn = func_id.split('::')[-1] if '::' in func_id else func_id
                    best = pd.get('best_without_hybrid', {})
                    line = f"  {fn}: textual={best.get('textual', 0):.1%} AST={best.get('structural', 0):.1%} behavioral_test={best.get('behavioral_test', 0):.1%}"
                    if pd.get('hybrid_loops_used', 0) > 0:
                        line += f" | hybrid_code%={pd.get('hybrid_code_percent', 0):.1f}%"
                    print(line)
                if len(paper_data) > 5:
                    print(f"  ... and {len(paper_data) - 5} more (see spec_results.json)")
            
            test_stats = analysis['test_statistics']
            print(f"\nTest Statistics:")
            print(f"  Tests generated: {test_stats['tests_generated']} functions")
            print(f"  Tests executed: {test_stats['tests_executed']} functions")
            print(f"  Total test cases: {test_stats.get('total_test_cases', 0)}")
            print(f"  Behavioral matches: {test_stats['behavioral_matches']}/{test_stats['tests_executed']}")
            print(f"  Behavioral match rate: {test_stats.get('behavioral_match_rate', 0):.1%}")
            if test_stats['tests_executed'] > 0:
                print(f"  Full branch coverage: {test_stats.get('full_branch_coverage', 0)}/{test_stats['tests_executed']} functions")
            
            if analysis['average_primary_similarity'] < 0.85:
                print(f"\nRecommendations:")
                print(f"  - Consider increasing max iterations for better results")
                print(f"  - Review failed functions for common patterns")
                print(f"  - Check if target similarity is achievable")
            elif analysis['average_primary_similarity'] >= 0.90:
                print(f"\nExcellent results! System is performing very well.")
            
            print(f"\nResults saved to: {args.output}")
            
        else:
            print(f"ERROR: Specification generation failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print(f"\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def extract_repo_name(repo_url: str) -> str:
    """Extract repository name from URL"""
    try:
        repo_url = repo_url.rstrip('.git')
        
        if 'github.com' in repo_url:
            parts = repo_url.split('/')
            if len(parts) >= 2:
                return f"{parts[-2]}-{parts[-1]}"
        
        return repo_url.split('/')[-1]
    
    except Exception:
        return "unknown-repo"


if __name__ == "__main__":
    main()

