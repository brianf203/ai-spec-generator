"""Quicksort."""
from partition import partition, partition_hoare
def quicksort_inplace(lst, lo=0, hi=None):
    hi = hi if hi is not None else len(lst) - 1
    if lo < hi:
        p = partition(lst, lo, hi)
        quicksort_inplace(lst, lo, p-1)
        quicksort_inplace(lst, p+1, hi)
    return lst
def quicksort_copy(lst):
    """Return sorted copy of lst."""
    return sorted(lst)
def quicksort_hoare(lst, lo=0, hi=None):
    hi = hi if hi is not None else len(lst) - 1
    if lo < hi:
        p = partition_hoare(lst, lo, hi)
        quicksort_hoare(lst, lo, p)
        quicksort_hoare(lst, p+1, hi)
    return lst
def quicksort_tail(lst, lo=0, hi=None):
    hi = hi if hi is not None else len(lst) - 1
    while lo < hi:
        p = partition(lst, lo, hi)
        if p - lo < hi - p: quicksort_tail(lst, lo, p-1); lo = p + 1
        else: quicksort_tail(lst, p+1, hi); hi = p - 1
    return lst
def select_kth(lst, k, lo=0, hi=None):
    hi = hi if hi is not None else len(lst) - 1
    if lo == hi: return lst[lo]
    p = partition(lst, lo, hi)
    if k == p: return lst[p]
    if k < p: return select_kth(lst, k, lo, p-1)
    return select_kth(lst, k, p+1, hi)