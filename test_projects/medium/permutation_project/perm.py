"""Permutation operations."""
def factorial(n):
    """Compute n!."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def perm_count(n, k):
    """Number of k-permutations of n."""
    if k > n:
        return 0
    result = 1
    for i in range(n - k + 1, n + 1):
        result *= i
    return result

def next_permutation(lst):
    """Next lexicographic permutation in-place. Returns False if last."""
    n = len(lst)
    i = n - 2
    while i >= 0 and lst[i] >= lst[i + 1]:
        i -= 1
    if i < 0:
        return False
    j = n - 1
    while lst[j] <= lst[i]:
        j -= 1
    lst[i], lst[j] = lst[j], lst[i]
    left, right = i + 1, n - 1
    while left < right:
        lst[left], lst[right] = lst[right], lst[left]
        left += 1
        right -= 1
    return True

def permute_list(lst):
    """Return all permutations of lst."""
    if len(lst) <= 1:
        return [list(lst)]
    result = []
    for i, x in enumerate(lst):
        rest = lst[:i] + lst[i+1:]
        for p in permute_list(rest):
            result.append([x] + p)
    return result