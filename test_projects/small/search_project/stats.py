"""Statistics from search results."""
def find_min(lst):
    """Return minimum value, or None if list is empty."""
    if not lst:
        return None
    result = lst[0]
    for i in range(1, len(lst)):
        v = lst[i]
        if v < result:
            result = v
    return result

def find_max(lst):
    """Return maximum value, or None if list is empty."""
    if not lst:
        return None
    result = lst[0]
    for i in range(1, len(lst)):
        v = lst[i]
        if v > result:
            result = v
    return result

def count_occurrences(lst, x):
    """Count how many times x appears in the list."""
    count = 0
    for v in lst:
        if v == x:
            count += 1
    return count

def sum_list(lst):
    """Return sum of all elements."""
    total = 0
    for v in lst:
        total += v
    return total

def average(lst):
    """Return arithmetic mean. Returns 0 for empty list."""
    if not lst:
        return 0
    total = sum_list(lst)
    count = len(lst)
    return total / count