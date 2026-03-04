"""Insertion sort."""
def insertion_sort(lst):
    arr = list(lst)
    for i in range(1, len(arr)):
        k = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > k:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = k
    return arr
def insertion_sort_desc(lst):
    """Sort descending using insertion sort."""
    return list(reversed(insertion_sort(lst)))

def insert_sorted(lst, x):
    """Insert x into sorted position in a copy of lst."""
    new_lst = list(lst)
    new_lst.append(x)
    new_lst.sort()
    return new_lst
def shell_sort(lst):
    arr, n = list(lst), len(lst)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            t, j = arr[i], i
            while j >= gap and arr[j-gap] > t: arr[j] = arr[j-gap]; j -= gap
            arr[j] = t
        gap //= 2
    return arr