# AI Spec Generator

A specification → regeneration → similarity pipeline for Python. Given real source code, it
generates a natural-language + structured **specification**, regenerates code from that spec with
**Anthropic Claude** (default `claude-sonnet-4-6`), and measures how close the regenerated code is
to the original along **structural** and **behavioral** axes.

The tool runs an iterative, multi-stage refinement loop and reports structural, behavioral, and
combined (primary) similarity separately.

---

## Pipeline

```
code_analyzer → spec_generator → code_regeneration → test_generation → test_execution
    → similarity_analyzer → feedback_loop → runtime_feedback_loop → convergence_checker
(repeat for N iterations)

Then, for functions still below the target similarity:
  Stage 2: failure_driven_refinement   (natural-language spec repair, no code)
  Stage 3: hybrid_specs                 (diagnostic, minimal code grafting)
```

---

## Components

- **Code analyzer** — parses the target files, extracts every callable in scope, resolves imports,
  and builds a dependency graph shared across functions.

- **Bounded context bundle** — for each target callable, assembles a deterministic, byte-budgeted
  excerpt pack: same-module callee bodies, the enclosing class envelope (`__init__` excerpt + sibling
  method headers), k-hop expansion across callees within a scoped subtree, and optional scoped name
  retrieval. Each chunk is recorded with a fingerprint for reproducibility.

- **Spec generator** — produces a natural-language summary plus a structured specification
  (signature, behavior, pre/post-conditions, error paths) using program slicing and slice-by-slice
  generation with causal prioritization.

- **Code regeneration** — regenerates an implementation from the specification alone.

- **Test generation / execution** — loads the repository's tests (or synthesizes a small behavioral
  harness) and runs them against both the original and regenerated code. When the AST loader finds no
  executable cases, a pytest subprocess oracle can run the repo's own suite, patch one function with
  the regenerated code, and score the pass/fail delta.

- **Similarity analyzer** — computes structural (AST), behavioral (test-based), and primary
  similarity. Markdown fences and prose are stripped before parsing so valid snippets are not
  penalized.

- **Feedback loops** — when similarity is below target, diff analysis and runtime test mismatches are
  fed back into the prompt for the next iteration.

- **Stage 2 — Failure-driven refinement** — infers missing spec content from the diff and adds
  natural-language updates (no code), then regenerates.

- **Stage 3 — Diagnostic hybrid** — when still below target, decomposes the gap (AST body diff,
  diff-line analysis, optional test-failure hints), ranks minimal original-code fragments, and adds
  them one at a time. A fragment is kept only if it improves similarity by at least the configured
  threshold; otherwise it is rejected and the next candidate is tried. Regeneration calls per function
  are capped.

- **Convergence checker** — stops once the target is met or the iteration/improvement budget is spent.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then edit and set ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY='your_key'
# Optional: CLAUDE_MAX_TOKENS (default 32768), CLAUDE_TEMPERATURE (default 0.2),
#           CLAUDE_MIN_INTERVAL (default 1.0)
```

---

## Usage

```bash
# Analyze a whole directory (must be a folder, not a single file)
python main.py --dir ./my_project -o ./output

# Scope to specific files, set target and iteration budget
python main.py --dir ./my_project --include src/pkg/module.py \
  -o ./output --target-similarity 0.90 --max-iterations 2

# Use the repository's own pytest suite as the behavioral oracle
python main.py --dir ./my_project --include src/pkg/module.py \
  --harness-mode auto --no-llm-tests -o ./output

# Stage ablations
python main.py --dir ./my_project -o ./out_no_slice  --no-slicing
python main.py --dir ./my_project -o ./out_no_hybrid --no-hybrid
```

---

## Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--dir` / `--repo` | — | Local directory or GitHub URL to analyze |
| `--include` | `*.py` | Files to include (relative globs/paths) |
| `--exclude` | tests, `__pycache__` | Files/dirs to exclude |
| `-o`, `--output` | `enhanced_output_v2` | Output directory |
| `--target-similarity` | 0.99 | Convergence target on primary similarity |
| `--max-iterations` | 10 | Iterations of the core loop |
| `--model` | `claude-sonnet-4-6` | Claude model id |
| `--harness-mode` | auto | `auto` \| `loader` \| `pytest` behavioral oracle |
| `--no-llm-tests` | off | Disable in-pipeline LLM test synthesis |
| `--no-slicing` / `--no-failure-driven` / `--no-hybrid` | off | Stage ablations |
| `--hybrid-similarity-threshold` | 0.99 | Similarity below which Stage 3 activates |
| `--hybrid-min-improvement` | 0.015 | Min similarity gain to keep a hybrid fragment |
| `--hybrid-max-regens-per-func` | 5 | Cap on hybrid regeneration calls per function |
| `--context-budget-chars` | 28672 | Byte budget for the bounded context bundle |

Run `python main.py --help` for the complete list; defaults also live in `config/default.py`.

---

## Output

Results are written to the output directory:

- `spec_results.json` — full results: per-function specifications, regenerated code, similarity
  metrics, phase tracking, and a `run_config` snapshot of the hyperparameters used.
- `specifications/` — per-function specifications.
- `generated_tests/` — generated/discovered test harness metadata.
- `test_results/` — per-function test execution results.

Phase tracking records which stage each function finished in: `normal`, `failure_driven`, or
`hybrid`. Metrics are reported separately as **structural**, **behavioral_test**, and **primary**.

---

## Project layout

```
main.py             CLI entry point (loads .env, builds the run config)
flow.py             SpecificationOrchestrator: iterations + Stage 2/3 wiring
nodes.py            Workflow nodes (analyze, spec, regen, test, similarity, hybrid, ...)
config/             Default hyperparameters
agents/             Spec generation, slicing, analysis, refinement, context bundle
utils/              LLM client, prompts, diff/gap analysis, pytest harness, test loader
requirements.txt    Python dependencies
.env.example        Environment template (set ANTHROPIC_API_KEY)
```
