"""Mergesort."""
from merge import merge_sorted_lists, merge_sorted_inplace
def mergesort(lst):
    if len(lst) <= 1:
        return lst
    mid = len(lst) // 2
    left = mergesort(lst[:mid])
    right = mergesort(lst[mid:])
    return merge_sorted_lists(left, right)
def mergesort_inplace(arr, lo=0, hi=None):
    hi = hi if hi is not None else len(arr) - 1
    if lo >= hi:
        return arr
    mid = (lo + hi) // 2
    mergesort_inplace(arr, lo, mid)
    mergesort_inplace(arr, mid + 1, hi)
    merge_sorted_inplace(arr, lo, mid, hi)
    return arr
def mergesort_iterative(lst):
    arr = list(lst)
    n = len(lst)
    size = 1
    while size < n:
        for lo in range(0, n - size, 2 * size):
            mid = lo + size - 1
            hi = min(lo + 2 * size - 1, n - 1)
            merge_sorted_inplace(arr, lo, mid, hi)
        size *= 2
    return arr