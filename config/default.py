"""
Default configuration for the specification generation system
"""

# LLM Configuration
LLM_MODEL = "claude-sonnet-4-6"
LLM_MIN_INTERVAL = 5.0  # Minimum seconds between API calls
LLM_MAX_RETRIES = 3
LLM_RATE_LIMIT_WAIT = 10  # Seconds to wait on rate limit errors

# Similarity Thresholds (100% AST + 100% behavioral tests, textual ignored)
DEFAULT_TARGET_SIMILARITY = 0.99
MAX_ITERATIONS = 10

# Three-stage pipeline (ablations: set any to False for experiments / baselines)
ENABLE_PROGRAM_SLICING = True  # False => monolithic-style spec (no slice-by-slice PDG path)
ENABLE_FAILURE_DRIVEN_REFINEMENT = True  # Stage 2: NL diff refinement
ENABLE_HYBRID_SPECS = True  # Stage 3: hybrid code additions
FAILURE_DRIVEN_MAX_ATTEMPTS = 3
HYBRID_MAX_ITERATIONS = 5
HYBRID_SIMILARITY_THRESHOLD = 0.99
MIN_IMPROVEMENT_FOR_EARLY_EXIT = 0.02  # Stop failure-driven if improvement < 2%

# Code Analysis
MAX_FILE_SIZE = 100000  # 100KB
INCLUDE_PATTERNS = ["*.py"]
EXCLUDE_PATTERNS = ["*test*", "tests/*", "__pycache__/*"]

# Test Generation
MIN_TESTS_PER_FUNCTION = 5
MAX_TESTS_PER_FUNCTION = 10
TARGET_BRANCH_COVERAGE = 0.80

# Similarity Metrics Weights
SIMILARITY_WEIGHTS = {
    'structural': 0.35,
    'behavioral_test': 0.25,
    'behavioral': 0.25,
    'semantic': 0.10,
    'textual': 0.05
}

