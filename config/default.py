"""
Default configuration for the specification generation system
"""

# LLM Configuration
LLM_MODEL = "gemini-2.0-flash"
LLM_MIN_INTERVAL = 5.0  # Minimum seconds between API calls
LLM_MAX_RETRIES = 3
LLM_RATE_LIMIT_WAIT = 10  # Seconds to wait on rate limit errors

# Similarity Thresholds (100% AST + 100% behavioral tests, textual ignored)
DEFAULT_TARGET_SIMILARITY = 1.0
MAX_ITERATIONS = 10

# Three-stage pipeline
FAILURE_DRIVEN_MAX_ATTEMPTS = 3
HYBRID_MAX_ITERATIONS = 5
HYBRID_SIMILARITY_THRESHOLD = 1.0
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

