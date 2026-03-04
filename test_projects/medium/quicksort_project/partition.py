"""Partition for quicksort."""
def partition(lst, lo, hi):
    pivot = lst[hi]
    i = lo - 1
    for j in range(lo, hi):
        if lst[j] <= pivot:
            i += 1
            lst[i], lst[j] = lst[j], lst[i]
    lst[i+1], lst[hi] = lst[hi], lst[i+1]
    return i + 1
def partition_first(lst, lo, hi):
    lst[lo], lst[hi] = lst[hi], lst[lo]
    return partition(lst, lo, hi)
def partition_mid(lst, lo, hi):
    mid = (lo + hi) // 2
    lst[mid], lst[hi] = lst[hi], lst[mid]
    return partition(lst, lo, hi)
def partition_hoare(lst, lo, hi):
    p, i, j = lst[lo], lo - 1, hi + 1
    while True:
        i += 1
        while lst[i] < p: i += 1
        j -= 1
        while lst[j] > p: j -= 1
        if i >= j: return j
        lst[i], lst[j] = lst[j], lst[i]
def three_way_partition(lst, lo, hi):
    lt, gt, i, v = lo, hi, lo, lst[lo]
    while i <= gt:
        if lst[i] < v: lst[lt], lst[i] = lst[i], lst[lt]; lt += 1; i += 1
        elif lst[i] > v: lst[i], lst[gt] = lst[gt], lst[i]; gt -= 1
        else: i += 1
    return lt, gt