# AI Spec Generator

Automated specification generation for Python code with a three-stage pipeline: normal generation, failure-driven refinement, and hybrid specs.

## Features

- **Stage 1 (Normal):** Program slicing, logical deletion, slice-by-slice spec generation with causal prioritization
- **Stage 2 (Failure-driven):** When similarity < threshold, analyzes diff to infer missing abstract spec content, adds natural language updates (no code snippets), regenerates
- **Stage 3 (Hybrid):** When still below threshold, incrementally adds code pieces until target similarity
- **Causal prioritization:** Uses causal inference to avoid deleting causally important code and to prioritize causally critical slices when merging

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY='your_key'
```

## Usage

```bash
# Analyze a project folder (must be a directory, not a single file)
python main.py --dir ./test_projects/small/calc_project --output ./output

# The system walks the folder, processes all .py files, skips READMEs and test files
python main.py --dir ./my_project --target-similarity 0.90 --max-iterations 5
```

## Test Projects

Realistic multi-file projects in `test_projects/`:

- **small/calc_project/**: operations.py, validation.py, constants.py, README.md, tests/
- **small/string_project/**: formatting.py, utils.py, README.md, tests/
- **medium/math_project/**: factorial.py, gcd_lcm.py, primes.py, digits.py, README.md, tests/
- **large/inventory_project/**: product.py, stock.py, reporting.py, utils.py, README.md, tests/

## Configuration

- `failure_driven_max_attempts`: 3 (Stage 2 attempts per function)
- `hybrid_max_iterations`: 5 (Stage 3 iterations per function)
- `min_improvement_for_early_exit`: 0.02 (Stop Stage 2 if improvement < 2%)

## Output

Results are saved to the output directory:
- `spec_results.json` - Full results with phase tracking
- `specifications/` - Per-function specifications
- `generated_tests/` - Generated test cases
- `test_results/` - Test execution results

Phase tracking records which stage succeeded for each function: `normal`, `failure_driven`, or `hybrid`.
