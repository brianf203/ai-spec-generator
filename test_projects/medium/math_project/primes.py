"""Primes."""
def is_prime(n):
    if n < 2:
        return False
    limit = int(n ** 0.5) + 1
    for i in range(2, limit):
        if n % i == 0:
            return False
    return True
def next_prime(n):
    c = n + 1
    while not is_prime(c):
        c += 1
    return c
def prev_prime(n):
    if n <= 2:
        return None
    c = n - 1
    while c >= 2 and not is_prime(c):
        c -= 1
    return c if c >= 2 else None
def nth_prime(n):
    c, p = 0, 2
    while c <= n:
        if is_prime(p):
            c += 1
        if c > n:
            return p
        p += 1
    return p
def prime_factors(n):
    out, d = [], 2
    while d * d <= n:
        while n % d == 0:
            out.append(d)
            n //= d
        d += 1
    if n > 1:
        out.append(n)
    return out
def count_primes_up_to(n):
    """Count primes in range [2, n] inclusive."""
    count = 0
    for i in range(2, n + 1):
        if is_prime(i):
            count += 1
    return count