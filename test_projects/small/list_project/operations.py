"""List operations."""
def _flatten(lst):
    out = []
    for x in lst:
        if isinstance(x, list):
            flattened = _flatten(x)
            for item in flattened:
                out.append(item)
        else:
            out.append(x)
    return out
def flatten_list(nested):
    """Flatten a nested list structure into a single-level list."""
    result = _flatten(nested)
    return result

def rotate_list(lst, n):
    """Rotate list left by n positions. Empty list returns empty."""
    if not lst:
        return []
    length = len(lst)
    shift = n % length
    if shift == 0:
        return list(lst)
    first_part = []
    for i in range(length - shift, length):
        first_part.append(lst[i])
    second_part = []
    for i in range(length - shift):
        second_part.append(lst[i])
    return first_part + second_part

def reverse_list(lst):
    """Return a new list with elements in reverse order."""
    result = []
    for i in range(len(lst) - 1, -1, -1):
        result.append(lst[i])
    return result

def first_n(lst, n):
    """Return the first n elements of the list."""
    if n <= 0:
        return []
    result = []
    for i in range(min(n, len(lst))):
        result.append(lst[i])
    return result

def last_n(lst, n):
    """Return the last n elements. Returns [] if n is 0."""
    if n == 0:
        return []
    start_idx = max(0, len(lst) - n)
    result = []
    for i in range(start_idx, len(lst)):
        result.append(lst[i])
    return result