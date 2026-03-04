"""Comparison and ordering utilities."""
def min_of_three(a, b, c):
    """Return minimum of three values."""
    result = a
    if b < result:
        result = b
    if c < result:
        result = c
    return result

def max_of_three(a, b, c):
    """Return maximum of three values."""
    result = a
    if b > result:
        result = b
    if c > result:
        result = c
    return result

def clamp_range(val, lo, hi):
    """Clamp val to [lo, hi]."""
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val

def in_range(val, lo, hi):
    """Check if val is in [lo, hi] inclusive."""
    return lo <= val <= hi

def is_sorted_asc(lst):
    """Check if list is sorted ascending."""
    for i in range(len(lst) - 1):
        if lst[i] > lst[i + 1]:
            return False
    return True

def is_sorted_desc(lst):
    """Check if list is sorted descending."""
    for i in range(len(lst) - 1):
        if lst[i] < lst[i + 1]:
            return False
    return True