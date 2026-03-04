"""Combination utilities."""
def comb_count(n, k):
    """Binomial coefficient C(n,k)."""
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result

def combine_list(lst, k):
    """All k-combinations of lst."""
    if k == 0:
        return [[]]
    if k > len(lst):
        return []
    result = []
    for i in range(len(lst) - k + 1):
        for c in combine_list(lst[i+1:], k - 1):
            result.append([lst[i]] + c)
    return result