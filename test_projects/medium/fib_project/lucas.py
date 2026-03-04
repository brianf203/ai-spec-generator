"""Lucas numbers."""
def lucas(n):
    if n == 0:
        return 2
    if n == 1:
        return 1
    a, b = 2, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
def lucas_list(n):
    """Return list of first n Lucas numbers."""
    result = []
    for i in range(n):
        result.append(lucas(i))
    return result
def fib_lucas_relation(n):
    from iterative import fibonacci_iterative
    return fibonacci_iterative(n) + lucas(n)