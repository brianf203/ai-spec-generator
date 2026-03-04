"""Bubble sort."""
def bubble_sort(lst):
    arr = list(lst)
    n = len(arr)
    for i in range(n):
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
def bubble_sort_desc(lst):
    """Sort descending using bubble sort."""
    return list(reversed(bubble_sort(lst)))

def bubble_sort_key(lst, key=None):
    """Sort by key function."""
    return sorted(lst, key=key)

def is_sorted(lst):
    """Check if list is non-decreasing."""
    if len(lst) <= 1:
        return True
    for i in range(len(lst) - 1):
        if lst[i] > lst[i + 1]:
            return False
    return True
def bubble_sort_inplace(lst):
    for i in range(len(lst)):
        for j in range(len(lst)-1-i):
            if lst[j] > lst[j+1]: lst[j], lst[j+1] = lst[j+1], lst[j]
    return lst