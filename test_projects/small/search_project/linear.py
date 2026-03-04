"""Linear search operations."""
def linear_search(lst, x):
    """Find index of first occurrence of x, or -1 if not found."""
    idx = -1
    for i, v in enumerate(lst):
        if v == x:
            idx = i
            break
    return idx

def contains(lst, x):
    """Check if x is in the list."""
    for v in lst:
        if v == x:
            return True
    return False

def find_all_indices(lst, x):
    """Return list of all indices where x appears."""
    result = []
    for i, v in enumerate(lst):
        if v == x:
            result.append(i)
    return result

def find_first(lst, pred):
    """Return first element satisfying pred, or None."""
    for v in lst:
        if pred(v):
            return v
    return None

def find_last(lst, pred):
    """Return last element satisfying pred, or None."""
    for i in range(len(lst) - 1, -1, -1):
        if pred(lst[i]):
            return lst[i]
    return None