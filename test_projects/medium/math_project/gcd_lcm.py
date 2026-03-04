"""GCD and LCM."""
def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)
def lcm(a, b):
    """Return least common multiple of a and b. Returns 0 if either is 0."""
    if not a or not b:
        return 0
    return abs(a * b) // gcd(a, b)

def gcd_recursive(a, b):
    """Compute GCD recursively using Euclidean algorithm."""
    if b == 0:
        return abs(a)
    return gcd_recursive(b, a % b)
def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    g, x1, y1 = extended_gcd(b % a, a)
    coeff = (b // a) * x1
    return g, y1 - coeff, x1
def lcm_many(nums):
    """Return LCM of a list of numbers."""
    return _lcm_reduce(nums)
def _lcm_reduce(lst):
    if len(lst) == 1:
        return lst[0]
    first = lst[0]
    rest = _lcm_reduce(lst[1:])
    return lcm(first, rest)