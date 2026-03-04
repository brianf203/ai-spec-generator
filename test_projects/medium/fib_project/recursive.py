"""Fibonacci recursive."""
def fibonacci_recursive(n):
    if n <= 1:
        return n
    a = fibonacci_recursive(n - 1)
    b = fibonacci_recursive(n - 2)
    return a + b
def fib_tail(n, a=0, b=1):
    """Tail-recursive Fibonacci."""
    if n == 0:
        return a
    return fib_tail(n - 1, b, a + b)
def fib_memo(n, memo=None):
    memo = memo or {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    a = fib_memo(n - 1, memo)
    b = fib_memo(n - 2, memo)
    memo[n] = a + b
    return memo[n]