"""Digit ops."""
def sum_digits(n):
    """Sum all digits of n (treating n as positive for the purpose)."""
    total = 0
    for d in str(abs(n)):
        total += int(d)
    return total
def product_digits(n):
    s = str(abs(n))
    p = 1
    for d in s:
        p *= int(d)
    return p
def digit_count(n):
    """Return number of digits in n."""
    s = str(abs(n))
    return len(s)
def reverse_digits(n):
    s = str(abs(n))
    return int(s[::-1]) if n >= 0 else -int(s[::-1])
def is_palindrome_number(n):
    """Check if n reads the same forwards and backwards."""
    s = str(n)
    return s == s[::-1]
def digit_root(n):
    while n >= 10:
        n = sum_digits(n)
    return n
def contains_digit(n, d):
    """Check if digit d appears in the decimal representation of n."""
    n_str = str(abs(n))
    d_str = str(d)
    return d_str in n_str