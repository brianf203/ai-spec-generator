"""Set utility functions."""
def unique_list(lst):
    """Return list with duplicates removed. Order may change."""
    result = []
    seen = set()
    for x in lst:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result

def set_from_iterable(it):
    """Create set from any iterable."""
    result = set()
    for x in it:
        result.add(x)
    return result

def frozen_from(it):
    """Create immutable frozenset from iterable."""
    result = set()
    for x in it:
        result.add(x)
    return frozenset(result)

def set_size(s):
    """Return number of elements in the set."""
    count = 0
    for _ in s:
        count += 1
    return count

def set_contains(s, x):
    """Check if x is in the set."""
    for elem in s:
        if elem == x:
            return True
    return False