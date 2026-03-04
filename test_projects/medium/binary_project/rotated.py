"""Rotated array search."""
def find_rotation_point(lst):
    lo, hi = 0, len(lst) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if lst[mid] > lst[hi]: lo = mid + 1
        else: hi = mid
    return lo
def search_rotated(lst, x):
    lo, hi = 0, len(lst) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if lst[mid] == x: return mid
        if lst[lo] <= lst[mid]:
            if lst[lo] <= x < lst[mid]: hi = mid - 1
            else: lo = mid + 1
        else:
            if lst[mid] < x <= lst[hi]: lo = mid + 1
            else: hi = mid - 1
    return -1
def min_in_rotated(lst):
    """Return minimum in rotated sorted list. None if empty."""
    if not lst:
        return None
    idx = find_rotation_point(lst)
    result = lst[idx]
    return result