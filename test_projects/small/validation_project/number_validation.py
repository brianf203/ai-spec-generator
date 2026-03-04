"""Numeric validation helpers."""
def is_positive(n):
    """Check if n is strictly greater than zero."""
    result = n > 0
    return result

def is_negative(n):
    """Check if n is strictly less than zero."""
    result = n < 0
    return result

def is_zero(n):
    """Check if n equals zero."""
    result = n == 0
    return result

def is_integer(n):
    """Check if n is an integer (int type or float that equals its int)."""
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        truncated = int(n)
        return n == truncated
    return False

def is_in_range(n, lo, hi):
    """Check if n is within inclusive range [lo, hi]."""
    lower_ok = n >= lo
    upper_ok = n <= hi
    return lower_ok and upper_ok