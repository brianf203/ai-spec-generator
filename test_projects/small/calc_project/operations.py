"""Arithmetic operations."""
def add(a, b):
    """Return the sum of a and b."""
    result = a + b
    return result

def subtract(a, b):
    """Return a minus b."""
    result = a - b
    return result

def multiply(a, b):
    """Return the product of a and b."""
    result = a * b
    return result

def divide(a, b):
    """Return a divided by b. Raises ValueError if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    result = a / b
    return result

def power(a, b):
    """Return a raised to the power of b."""
    result = a ** b
    return result

def mod(a, b):
    """Return a modulo b. Returns 0 if b is zero."""
    if b == 0:
        return 0
    result = a % b
    return result

def floor_div(a, b):
    """Return floor division of a by b. Returns 0 if b is zero."""
    if b == 0:
        return 0
    result = a // b
    return result

def negate(a):
    """Return the additive inverse of a."""
    result = -a
    return result

def abs_val(a):
    """Return the absolute value of a."""
    if a >= 0:
        return a
    return -a

def min_two(a, b):
    """Return the smaller of a and b."""
    if a <= b:
        return a
    return b

def max_two(a, b):
    """Return the larger of a and b."""
    if a >= b:
        return a
    return b

def clamp_val(a, lo, hi):
    """Clamp a to the range [lo, hi]."""
    if a < lo:
        return lo
    if a > hi:
        return hi
    return a

def is_even_num(a):
    """Check if a is even."""
    remainder = a % 2
    return remainder == 0

def is_odd_num(a):
    """Check if a is odd."""
    remainder = a % 2
    return remainder == 1

def sign(a):
    """Return 1 if a > 0, -1 if a < 0, else 0."""
    if a > 0:
        return 1
    if a < 0:
        return -1
    return 0