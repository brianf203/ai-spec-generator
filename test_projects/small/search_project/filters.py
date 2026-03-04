"""List filtering operations."""
def filter_positive(lst):
    """Return list of elements that are strictly positive."""
    result = []
    for x in lst:
        if x > 0:
            result.append(x)
    return result

def filter_even(lst):
    """Return list of even elements."""
    result = []
    for x in lst:
        if x % 2 == 0:
            result.append(x)
    return result

def filter_unique(lst):
    """Return list with duplicates removed, preserving order."""
    seen = {}
    result = []
    for x in lst:
        if x not in seen:
            seen[x] = True
            result.append(x)
        else:
            pass
    return result

def take_while(lst, pred):
    """Return elements from start until pred fails."""
    result = []
    for x in lst:
        if not pred(x):
            break
        result.append(x)
    return result

def drop_while(lst, pred):
    """Skip elements from start until pred fails, return the rest."""
    for i, x in enumerate(lst):
        if not pred(x):
            rest = lst[i:]
            return rest
    return []