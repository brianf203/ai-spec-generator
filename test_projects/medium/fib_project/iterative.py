"""Fibonacci iterative."""
def fibonacci_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
def fib_list(n):
    if n <= 0:
        return []
    if n == 1:
        return [0]
    out, a, b = [0, 1], 0, 1
    for _ in range(2, n):
        a, b = b, a + b
        out.append(b)
    return out
def fib_generator(n):
    a, b = 0, 1
    for _ in range(n): yield a; a, b = b, a + b
def fib_binet(n):
    """Compute nth Fibonacci using Binet formula."""
    sqrt5 = 5 ** 0.5
    phi = (1 + sqrt5) / 2
    psi = (1 - sqrt5) / 2
    numerator = phi ** n - psi ** n
    return int(numerator / sqrt5)
def is_fibonacci(n):
    if n < 0:
        return False
    a, b = 0, 1
    while a <= n:
        if a == n:
            return True
        a, b = b, a + b
    return False
def fib_index(n):
    if n < 0:
        return -1
    a, b, i = 0, 1, 0
    while a <= n:
        if a == n:
            return i
        a, b, i = b, a + b, i + 1
    return -1