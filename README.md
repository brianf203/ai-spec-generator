# AI-Powered Code Specification Generator

An intelligent system that generates comprehensive specifications from Python code and validates them through automated testing and code regeneration.

## Features

- **Automated Specification Generation**: Analyzes Python code using AST and generates detailed specifications
- **Code Regeneration**: Regenerates code from specifications only (no access to original)
- **Automated Test Generation**: Creates 5-10 test cases per function for behavioral validation
- **Multi-Metric Similarity Analysis**: 
  - Structural (AST): 35%
  - Behavioral tests: 25%
  - Behavioral patterns: 25%
  - Semantic: 10%
  - Textual: 5%
- **Dual Feedback Loops**: Iterative refinement through prompt modification and test failure accumulation
- **No Hardcoding**: Works on any Python project without customization

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-spec-generator.git
cd ai-spec-generator

# Install requirements
pip install -r requirements.txt
```

## Usage

```bash
# Set your Gemini API key
export GEMINI_API_KEY='your_api_key_here'

# Analyze a local directory
python main.py --dir /path/to/python/project

# With custom settings
python main.py \
  --dir /path/to/project \
  --target-similarity 0.90 \
  --max-iterations 3 \
  --output results \
  --verbose
```

## Example

```bash
# Quick test on the included sample project
python main.py \
  --dir test_project \
  --target-similarity 0.90 \
  --max-iterations 3 \
  --output demo_results \
  --api-key YOUR_API_KEY
```

## Results

Typical performance on different project sizes:

- **Small projects (10 functions)**: 93% similarity, 2 minutes
- **Medium projects (99 functions)**: 83% similarity, 30 minutes
- **Large projects (500+ functions)**: Processing in progress

## Requirements

- Python 3.8+
- Google Gemini API key
- See `requirements.txt` for full dependencies

## Architecture

The system follows a workflow:
1. **AST Analysis** - Parse code structure
2. **Specification Generation** - LLM creates detailed specs
3. **Code Regeneration** - Regenerate from specs only
4. **Test Generation** - Create automated tests
5. **Test Execution** - Run tests on both versions
6. **Similarity Analysis** - Multi-metric comparison
7. **Feedback Loops** - Iterative refinement

## License

MIT License

