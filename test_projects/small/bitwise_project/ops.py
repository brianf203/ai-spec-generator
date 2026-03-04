"""Bitwise operations."""
def bit_and(a, b):
    """Return bitwise AND of a and b."""
    return a & b

def bit_or(a, b):
    """Return bitwise OR of a and b."""
    return a | b

def bit_xor(a, b):
    """Return bitwise XOR of a and b."""
    return a ^ b

def bit_not(a):
    """Return bitwise NOT (complement) of a."""
    return ~a

def left_shift(a, n):
    """Shift a left by n bits."""
    return a << n

def right_shift(a, n):
    """Shift a right by n bits."""
    return a >> n

def count_ones(n):
    """Count number of 1 bits in n."""
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count

def is_power_of_two(n):
    """Check if n is a power of 2."""
    if n <= 0:
        return False
    return (n & (n - 1)) == 0