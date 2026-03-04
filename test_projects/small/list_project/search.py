"""List search and index operations."""
def index_of(lst, x):
    """Return index of first occurrence of x, or -1 if not found."""
    for i, v in enumerate(lst):
        if v == x:
            return i
    return -1

def count_val(lst, x):
    """Count how many times x appears in the list."""
    count = 0
    for v in lst:
        if v == x:
            count += 1
    return count

def contains(lst, x):
    """Check if x is in the list."""
    for v in lst:
        if v == x:
            return True
    return False

def min_idx(lst):
    """Return index of minimum element. Returns -1 for empty list."""
    if not lst:
        return -1
    min_val = lst[0]
    min_i = 0
    for i in range(1, len(lst)):
        if lst[i] < min_val:
            min_val = lst[i]
            min_i = i
    return min_i

def max_idx(lst):
    """Return index of maximum element. Returns -1 for empty list."""
    if not lst:
        return -1
    max_val = lst[0]
    max_i = 0
    for i in range(1, len(lst)):
        if lst[i] > max_val:
            max_val = lst[i]
            max_i = i
    return max_i