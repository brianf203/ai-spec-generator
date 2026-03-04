"""Power and root operations for numbers."""
def square(n):
    """Compute the square of n."""
    result = n * n
    return result

def cube(n):
    """Compute the cube of n."""
    squared = n * n
    result = squared * n
    return result

def abs_value(n):
    """Return the absolute value of n."""
    if n >= 0:
        result = n
    else:
        result = -n
    return result

def quad(n):
    """Compute n raised to the fourth power."""
    squared = n * n
    result = squared * squared
    return result

def sqrt_approx(n):
    """Return integer approximation of square root. Returns 0 for negative n."""
    if n < 0:
        return 0
    approx = n ** 0.5
    result = int(approx)
    return result