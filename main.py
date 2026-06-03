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
    parser.add_argument(
        "--only-qualified-name",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Only analyze functions whose qualified AST key matches (e.g. YoutubeIE._real_extract or foo). "
             "Repeat flag or pass multiple names. Omit to analyze all functions matching include/exclude.",
    )
    parser.add_argument("-e", "--exclude", nargs="*", default=["*test*", "test_*", "tests/*", "__pycache__/*"], help="Files/dirs to exclude (e.g. *test*, test_*, tests/*)")
    parser.add_argument("-s", "--max-size", type=int, default=100000, help="Maximum file size in bytes (default: 100KB)")
    parser.add_argument("--target-similarity", type=float, default=0.99,
                        help="Convergence target on primary similarity (default: 0.99; see paper)")
    parser.add_argument("--hybrid-similarity-threshold", type=float, default=0.99,
                        help="Hybrid Stage 3 activates when primary similarity is below this (default: 0.99)")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum iterations (default: 10)")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY environment variable)")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Claude model id (default: claude-sonnet-4-6)")
    parser.add_argument("--no-cache", action="store_true", help="Disable LLM response caching")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    ablation = parser.add_argument_group("Ablations / baselines (for experiments)")
    ablation.add_argument("--no-slicing", action="store_true",
                          help="Baseline: disable PDG slice-by-slice spec path (monolithic-style generation)")
    ablation.add_argument("--no-failure-driven", action="store_true",
                          help="Disable Stage 2 (failure-driven NL refinement)")
    ablation.add_argument("--no-hybrid", action="store_true",
                          help="Disable Stage 3 (hybrid code additions)")
    ablation.add_argument(
        "--hybrid-min-improvement",
        type=float,
        default=0.015,
        help="Minimum primary-similarity gain per hybrid snippet to keep it (default: 0.015).",
    )
    ablation.add_argument(
        "--hybrid-max-regens-per-func",
        type=int,
        default=5,
        metavar="N",
        help="Max regeneration API calls during diagnostic hybrid per function (default: 5).",
    )
    ablation.add_argument(
        "--hybrid-allow-full-code-fallback",
        action="store_true",
        help="Allow legacy last-resort: paste entire original function into spec (default: off).",
    )
    ctx = parser.add_argument_group("Bounded context bundle (v2)")
    ctx.add_argument(
        "--no-context-bundle",
        action="store_true",
        help="Disable deterministic bounded-context packaging (same-module / class envelope / scoped k-hop).",
    )
    ctx.add_argument(
        "--context-budget-chars",
        type=int,
        default=28672,
        metavar="N",
        help="Total UTF-8 budget when assembling dependency excerpts (default: 28672).",
    )
    ctx.add_argument(
        "--context-k-hop",
        type=int,
        default=2,
        metavar="K",
        help="Callee expansion depth within scoped subtree (default: 2).",
    )
    ctx.add_argument(
        "--context-scope-parent-levels",
        type=int,
        default=1,
        metavar="L",
        help="Ancestor directory roots upward from anchor file (within project): 0=anchor dir only (default: 1).",
    )
    ctx.add_argument(
        "--context-enable-rag",
        action="store_true",
        help="Include name-index fallback within scope when callees are unresolved (deterministic retrieval).",
    )
    ctx.add_argument(
        "--context-spec-prompt-bytes",
        type=int,
        default=16384,
        metavar="N",
        help="Max UTF-8 bytes of bundle text appended to specification-generation prompts (default: 16384).",
    )
    ctx.add_argument(
        "--context-regen-bundle-bytes",
        type=int,
        default=24576,
        metavar="N",
        help="Max UTF-8 bytes of bounded_context_bundle stored in specification for regeneration (default: 24576).",
    )
    ctx.add_argument(
        "--context-test-prompt-bundle-bytes",
        type=int,
        default=12288,
        metavar="N",
        help="Max UTF-8 bytes of bounded_context_bundle inlined into LLM test synthesis prompts (default: 12288).",
    )
    tst = parser.add_argument_group("Behavioral harness")
    tst.add_argument(
        "--no-llm-tests",
        action="store_true",
        help="Disable synthesizing parameterized harness tests when repo unittest extraction finds none or too few cases.",
    )
    tst.add_argument(
        "--min-behavioral-cases",
        type=int,
        default=3,
        metavar="N",
        help="Minimum executed harness cases required before averaging AST + behavioral into primary similarity (default: 3).",
    )
    tst.add_argument(
        "--max-llm-test-rounds",
        type=int,
        default=2,
        metavar="R",
        help="Max LLM test-synthesis batches per function across hybrid/regen reruns (default: 2).",
    )
    tst.add_argument(
        "--harness-mode",
        choices=("auto", "loader", "pytest"),
        default="auto",
        help=(
            "Behavioral oracle: auto=AST loader then pytest if empty; "
            "loader=AST only; pytest=subprocess pytest on discovered test modules (default: auto)."
        ),
    )
    tst.add_argument(
        "--pytest-harness-timeout-sec",
        type=int,
        default=300,
        metavar="SEC",
        help="Timeout per pytest subprocess invocation in pytest harness mode (default: 300).",
    )
    met = parser.add_argument_group("Metrics / regeneration fidelity")
    met.add_argument(
        "--regeneration-spec-json-char-budget",
        type=int,
        default=56000,
        metavar="N",
        help="Approx. character ceiling for specification JSON pasted into codegen prompts (default: 56000).",
    )
    met.add_argument(
        "--no-trusted-oracle-blend",
        action="store_true",
        help="Disable trusted-harness uplift for primary_similarity once min behavioral cases execute.",
    )
    met.add_argument(
        "--trusted-behavioral-agreement-floor",
        type=float,
        default=0.999,
        help="Harness agreement threshold to activate oracle blending (default: 0.999).",
    )
    met.add_argument(
        "--behavioral-oracle-blend-weight",
        type=float,
        default=0.72,
        help="Oracle blend coefficient toward behavioral correctness when floor is met (default: 0.72).",
    )

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
    
    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not provided")
        print("   Set environment variable: export ANTHROPIC_API_KEY='your_key'")
        print("   Or use --api-key argument")
        sys.exit(1)
    
    os.environ["ANTHROPIC_API_KEY"] = api_key
    
    from utils.call_llm import init_llm_from_config
    enable_slicing = not args.no_slicing
    enable_fd = not args.no_failure_driven
    enable_hybrid = not args.no_hybrid
    config = {
        'api_key': api_key,
        'model': args.model,
        'target_similarity': args.target_similarity,
        'max_iterations': args.max_iterations,
        'max_file_size': args.max_size,
        'include_patterns': args.include,
        'exclude_patterns': args.exclude,
        'only_qualified_names': list(args.only_qualified_name) if args.only_qualified_name else None,
        'output_dir': args.output,
        'verbose': args.verbose,
        'cache_enabled': not args.no_cache,
        'enable_program_slicing': enable_slicing,
        'enable_failure_driven_refinement': enable_fd,
        'enable_hybrid_specs': enable_hybrid,
        'failure_driven_max_attempts': 3,
        'hybrid_max_iterations': 12,
        'min_improvement_for_early_exit': 0.02,
        'hybrid_similarity_threshold': args.hybrid_similarity_threshold,
        'hybrid_min_improvement_per_step': args.hybrid_min_improvement,
        'hybrid_max_regens_per_func': args.hybrid_max_regens_per_func,
        'hybrid_allow_full_code_fallback': args.hybrid_allow_full_code_fallback,
        'enable_context_bundle': (not args.no_context_bundle),
        'context_budget_chars': args.context_budget_chars,
        'context_k_hop': args.context_k_hop,
        'context_scope_parent_levels': args.context_scope_parent_levels,
        'context_enable_rag': args.context_enable_rag,
        'context_spec_prompt_inject_chars': args.context_spec_prompt_bytes,
        'context_regen_bundle_chars': args.context_regen_bundle_bytes,
        'enable_llm_generated_tests': (not args.no_llm_tests),
        'min_behavioral_cases': args.min_behavioral_cases,
        'max_llm_test_generation_rounds_per_func': args.max_llm_test_rounds,
        'harness_mode': args.harness_mode,
        'pytest_harness_timeout_sec': args.pytest_harness_timeout_sec,
        'context_test_prompt_bundle_bytes': args.context_test_prompt_bundle_bytes,
        'regeneration_spec_json_char_budget': args.regeneration_spec_json_char_budget,
        'trusted_behavioral_oracle_blend': (not args.no_trusted_oracle_blend),
        'trusted_behavioral_agreement_floor': args.trusted_behavioral_agreement_floor,
        'behavioral_oracle_blend_weight': args.behavioral_oracle_blend_weight,
    }
    init_llm_from_config(config)
    
    print("Testing LLM connection...")
    if not test_llm_connection():
        print("ERROR: Failed to connect to Anthropic Claude API")
        print("   Please check ANTHROPIC_API_KEY and internet connection")
        sys.exit(1)
    
    print("Connected to Anthropic Claude API")
    
    if args.repo:
        project_path = args.repo
        project_name = args.name or extract_repo_name(args.repo)
    else:
        project_path = args.dir
        project_name = args.name or os.path.basename(os.path.abspath(args.dir))
    
    print(f"\n{'='*70}")
    print(f"Project: {project_name}")
    print(f"Target similarity: {args.target_similarity:.1%}")
    print(f"Hybrid similarity threshold: {args.hybrid_similarity_threshold:.1%}")
    print(f"Max iterations: {args.max_iterations}")
    if args.only_qualified_name:
        print(f"Only qualified names: {', '.join(args.only_qualified_name)}")
    print(f"Ablations: slicing={enable_slicing} failure_driven={enable_fd} hybrid={enable_hybrid}")
    print(
        f"Context bundle: enabled={not args.no_context_bundle} "
        f"budget={args.context_budget_chars} k_hop={args.context_k_hop} "
        f"scope_parents={args.context_scope_parent_levels} rag={args.context_enable_rag}"
    )
    print(
        f"Behavioral harness: mode={args.harness_mode} llm_tests={not args.no_llm_tests} "
        f"min_cases_for_blend={args.min_behavioral_cases} llm_rounds_cap={args.max_llm_test_rounds} "
        f"pytest_timeout={args.pytest_harness_timeout_sec}s"
    )
    print(
        f"Metrics: regen_spec_json_budget={args.regeneration_spec_json_char_budget} "
        f"trusted_oracle_blend={not args.no_trusted_oracle_blend} "
        f"oracle_floor={args.trusted_behavioral_agreement_floor} "
        f"oracle_alpha={args.behavioral_oracle_blend_weight}"
    )
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
            rc = results.get('run_config') or analysis.get('run_config')
            if rc:
                print(f"\nRun configuration (saved to spec_results.json): {rc}")
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
            print(f"    (same as test-case agreement: matching orig vs regen outcomes / all generated test cases)")
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
                s2 = loop_stats.get('stage2_after_feedback', {})
                s3 = loop_stats.get('stage3_hybrid', {})
                p1 = 0.0
                if s1.get('functions', 0) > 0:
                    p1 = s1.get('avg_primary', 0) or s1.get('avg_structural', 0)
                    print(f"  Stage 1 (first run, no feedback/hybrid): primary={p1:.1%} textual={s1.get('avg_textual', 0):.1%} AST={s1.get('avg_structural', 0):.1%} behavioral_test={s1.get('avg_behavioral_test', 0):.1%}")
                if s2.get('functions', 0) > 0:
                    p2 = s2.get('avg_primary', 0) or s2.get('avg_structural', 0)
                    print(f"  Stage 2 (after feedback loops):         primary={p2:.1%} textual={s2.get('avg_textual', 0):.1%} AST={s2.get('avg_structural', 0):.1%} behavioral_test={s2.get('avg_behavioral_test', 0):.1%}")
                    if s1.get('functions', 0) > 0:
                        delta = p2 - p1
                        print(f"  Stage 2 improvement: +{delta:.1%} primary similarity from Stage 1")
                if s3.get('functions_using_hybrid', 0) > 0:
                    print(f"  Stage 3 (hybrid code % to reach 100%):   {s3.get('functions_using_hybrid', 0)} functions, avg={s3.get('avg_code_percent', 0):.1f}% min={s3.get('min_code_percent', 0):.1f}% max={s3.get('max_code_percent', 0):.1f}%")

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
            tc_match = test_stats.get('test_cases_matching', 0)
            tc_total = test_stats.get('total_test_cases', 0)
            tc_rate = test_stats.get('test_case_agreement_rate', 0)
            print(f"  Test-case agreement: {tc_match}/{tc_total} cases = {tc_rate:.1%} (orig vs regen same outcome)")
            print(f"  Behavioral matches (all tests pass, per function): {test_stats['behavioral_matches']}/{test_stats['tests_executed']}")
            print(f"  Behavioral match rate (strict per-function): {test_stats.get('behavioral_match_rate', 0):.1%}")
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

