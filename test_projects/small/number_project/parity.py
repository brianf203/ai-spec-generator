"""Parity and divisibility checks for integers."""
def is_even(n):
    """Check if n is divisible by 2."""
    remainder = n % 2
    return remainder == 0

def is_odd(n):
    """Check if n has remainder 1 when divided by 2."""
    remainder = n % 2
    return remainder == 1

def is_divisible(n, d):
    """Check if n is evenly divisible by d. Returns False if d is zero."""
    if d == 0:
        return False
    remainder = n % d
    return remainder == 0

def next_even(n):
    """Return the smallest even number >= n."""
    remainder = n % 2
    if remainder == 0:
        return n
    return n + 1

def prev_odd(n):
    """Return the largest odd number <= n."""
    remainder = n % 2
    if remainder == 1:
        return n
    return n - 1