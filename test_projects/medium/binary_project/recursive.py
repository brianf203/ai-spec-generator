"""Binary search recursive."""
def binary_search_recursive(lst, x, lo=0, hi=None):
    hi = hi if hi is not None else len(lst) - 1
    if lo > hi:
        return -1
    mid = (lo + hi) // 2
    if lst[mid] == x:
        return mid
    if lst[mid] < x:
        return binary_search_recursive(lst, x, mid + 1, hi)
    return binary_search_recursive(lst, x, lo, mid - 1)
def bsearch_first_ge(lst, x, lo=0, hi=None):
    hi = hi if hi is not None else len(lst)
    if lo >= hi: return lo if lo < len(lst) else -1
    mid = (lo + hi) // 2
    if lst[mid] < x: return bsearch_first_ge(lst, x, mid+1, hi)
    return bsearch_first_ge(lst, x, lo, mid)
def bsearch_last_le(lst, x, lo=0, hi=None):
    hi = hi if hi is not None else len(lst)
    if lo >= hi: return hi - 1 if hi > 0 else -1
    mid = (lo + hi) // 2
    if lst[mid] <= x: return bsearch_last_le(lst, x, mid+1, hi)
    return bsearch_last_le(lst, x, lo, mid)