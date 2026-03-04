"""Binary search."""
def binary_search(lst, x):
    lo, hi = 0, len(lst) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if lst[mid] == x:
            return mid
        if lst[mid] < x:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
def binary_search_left(lst, x):
    lo, hi = 0, len(lst)
    while lo < hi:
        mid = (lo + hi) // 2
        if lst[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(lst) and lst[lo] == x:
        return lo
    return -1
def binary_search_right(lst, x):
    lo, hi = 0, len(lst)
    while lo < hi:
        mid = (lo + hi) // 2
        if lst[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    if lo > 0 and lst[lo - 1] == x:
        return lo - 1
    return -1
def binary_search_closest(lst, x):
    if not lst:
        return -1
    lo, hi = 0, len(lst) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if lst[mid] <= x:
            lo = mid
        else:
            hi = mid
    dist_lo = abs(lst[lo] - x)
    dist_hi = abs(lst[hi] - x)
    return lo if dist_lo <= dist_hi else hi
def binary_search_range(lst, x):
    left = binary_search_left(lst, x)
    right = binary_search_right(lst, x)
    return (left, right)
def is_in_sorted(lst, x):
    idx = binary_search(lst, x)
    return idx >= 0