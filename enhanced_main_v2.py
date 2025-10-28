"""
Enhanced Main Entry Point V2
Command-line interface for the enhanced specification generation system
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, Optional
from pathlib import Path

from enhanced_flow_v2 import create_enhanced_pocketflow_orchestrator
from utils.call_llm import test_llm_connection


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced PocketFlow-based Code Specification Generator with Test Generation",
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
    parser.add_argument("-e", "--exclude", nargs="*", default=["*test*", "tests/*", "__pycache__/*"], help="Files to exclude")
    parser.add_argument("-s", "--max-size", type=int, default=100000, help="Maximum file size in bytes (default: 100KB)")
    parser.add_argument("--target-similarity", type=float, default=0.95, help="Target similarity threshold (default: 0.95)")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum iterations (default: 10)")
    parser.add_argument("--api-key", help="Gemini API key (or set GEMINI_API_KEY environment variable)")
    parser.add_argument("--model", default="gemini-2.0-flash-exp", help="Gemini model (default: gemini-2.0-flash-exp)")
    parser.add_argument("--no-cache", action="store_true", help="Disable LLM response caching")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.repo and not args.repo.startswith(('http://', 'https://')):
        print("❌ Repository URL must start with http:// or https://")
        sys.exit(1)
    
    if args.dir and not os.path.exists(args.dir):
        print(f"❌ Directory not found: {args.dir}")
        sys.exit(1)
    
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ GEMINI_API_KEY not provided")
        print("   Set environment variable: export GEMINI_API_KEY='your_key'")
        print("   Or use --api-key argument")
        sys.exit(1)
    
    os.environ["GEMINI_API_KEY"] = api_key
    
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
        'cache_enabled': not args.no_cache
    }
    
    print("🧪 Testing LLM connection...")
    if not test_llm_connection():
        print("❌ Failed to connect to Gemini API")
        print("   Please check your API key and internet connection")
        sys.exit(1)
    
    print("✅ Connected to Gemini API")
    
    if args.repo:
        project_path = args.repo
        project_name = args.name or extract_repo_name(args.repo)
    else:
        project_path = args.dir
        project_name = args.name or os.path.basename(os.path.abspath(args.dir))
    
    print(f"\n{'='*70}")
    print(f"📁 Project: {project_name}")
    print(f"🎯 Target similarity: {args.target_similarity:.1%}")
    print(f"🔄 Max iterations: {args.max_iterations}")
    print(f"🧪 Features: Test generation + Dual feedback loops")
    print(f"{'='*70}")
    
    os.makedirs(args.output, exist_ok=True)
    
    print("\n🚀 Initializing Enhanced PocketFlow orchestrator...")
    orchestrator = create_enhanced_pocketflow_orchestrator(config)
    
    start_time = time.time()
    
    try:
        results = orchestrator.process_project(project_path, args.target_similarity)
        end_time = time.time()
        
        orchestrator.save_results(results, args.output)
        
        if results['success']:
            analysis = results['analysis']
            print(f"\n{'='*70}")
            print(f"🎉 Specification generation completed!")
            print(f"{'='*70}")
            print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
            print(f"\n📊 Function Statistics:")
            print(f"  Total functions: {analysis['total_functions']}")
            print(f"  ✅ Successful: {analysis['successful_functions']}")
            print(f"  ❌ Failed: {analysis['failed_functions']}")
            print(f"\n📈 Similarity Metrics:")
            print(f"  Average similarity: {analysis['average_similarity']:.1%}")
            print(f"  🎯 Success rate: {analysis['success_rate']:.1%}")
            print(f"  Target achieved: {analysis['target_achieved_count']}/{analysis['total_functions']} functions")
            print(f"\n🔄 Iteration Metrics:")
            print(f"  Iterations completed: {analysis['iterations_completed']}")
            print(f"  🎯 Convergence achieved: {analysis['convergence_achieved']}")
            
            test_stats = analysis['test_statistics']
            print(f"\n🧪 Test Statistics:")
            print(f"  Tests generated: {test_stats['tests_generated']} functions")
            print(f"  Tests executed: {test_stats['tests_executed']} functions")
            print(f"  Total test cases: {test_stats.get('total_test_cases', 0)}")
            print(f"  Behavioral matches: {test_stats['behavioral_matches']}/{test_stats['tests_executed']}")
            print(f"  🎯 Behavioral match rate: {test_stats.get('behavioral_match_rate', 0):.1%}")
            
            print(f"\n📊 Similarity Distribution:")
            for category, count in analysis['similarity_distribution'].items():
                print(f"  {category}: {count} functions")
            
            if analysis['average_similarity'] < 0.85:
                print(f"\n💡 Recommendations:")
                print(f"  - Consider increasing max iterations for better results")
                print(f"  - Review failed functions for common patterns")
                print(f"  - Check if target similarity is achievable")
            elif analysis['average_similarity'] >= 0.90:
                print(f"\n🎉 Excellent results! System is performing very well.")
            
            print(f"\n📁 Results saved to: {args.output}")
            
        else:
            print(f"❌ Specification generation failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
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

