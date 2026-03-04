"""Selection sort."""
def selection_sort(lst):
    arr = list(lst)
    n = len(arr)
    for i in range(n):
        mi = i
        for j in range(i + 1, n):
            if arr[j] < arr[mi]:
                mi = j
        arr[i], arr[mi] = arr[mi], arr[i]
    return arr
def selection_sort_desc(lst):
    """Sort descending using selection sort."""
    return list(reversed(selection_sort(lst)))

def selection_sort_key(lst, key=None):
    """Sort by key function."""
    return sorted(lst, key=key)

def min_index(lst, start=0):
    """Return index of minimum in lst[start:]. Returns -1 if slice is empty."""
    sub = lst[start:]
    if not sub:
        return -1
    m = min(sub)
    for i, v in enumerate(sub):
        if v == m:
            return start + i
    return -1

def max_index(lst, start=0):
    """Return index of maximum in lst[start:]. Returns -1 if slice is empty."""
    sub = lst[start:]
    if not sub:
        return -1
    m = max(sub)
    for i, v in enumerate(sub):
        if v == m:
            return start + i
    return -1
def selection_sort_inplace(lst):
    for i in range(len(lst)):
        mi = min(range(i, len(lst)), key=lambda j: lst[j])
        lst[i], lst[mi] = lst[mi], lst[i]
    return lst