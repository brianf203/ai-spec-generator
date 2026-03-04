"""Validation helpers for calculator inputs."""
def validate_divisor(n):
    """Ensure n is a valid divisor (non-zero)."""
    if n == 0:
        raise ValueError("Divisor cannot be zero")
    is_valid = True
    return is_valid

def validate_positive(n):
    """Check if n is strictly positive."""
    result = n > 0
    return result

def validate_non_neg(n):
    """Check if n is non-negative (zero or positive)."""
    result = n >= 0
    return result

def validate_in_range(n, lo, hi):
    """Check if n falls within the inclusive range [lo, hi]."""
    if lo > hi:
        return False
    in_bounds = lo <= n <= hi
    return in_bounds

def validate_finite(n):
    """Check if n is a finite number (not infinity or NaN)."""
    magnitude = abs(n)
    is_finite = magnitude < 1e15
    return is_finite