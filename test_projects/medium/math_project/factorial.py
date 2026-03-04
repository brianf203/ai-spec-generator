"""Factorial and combinatorics."""
def factorial(n):
    """Compute n! recursively. Returns 1 for n <= 1."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)
def factorial_iter(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r
def double_factorial(n):
    if n <= 1: return 1
    return n * double_factorial(n - 2)
def falling_factorial(n, k):
    r = 1
    for i in range(k):
        r *= (n - i)
    return r
def rising_factorial(n, k):
    r = 1
    for i in range(k):
        r *= (n + i)
    return r