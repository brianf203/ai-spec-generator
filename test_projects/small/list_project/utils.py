"""List utility functions."""
def chunk_list(lst, size):
    """Split list into chunks of given size."""
    result = []
    for i in range(0, len(lst), size):
        chunk = lst[i:i + size]
        result.append(chunk)
    return result

def zip_lists(a, b):
    """Combine two lists into list of pairs. Stops at shorter length."""
    result = []
    for i in range(min(len(a), len(b))):
        result.append((a[i], b[i]))
    return result

def sum_list(lst):
    """Return the sum of all elements in the list."""
    total = 0
    for x in lst:
        total += x
    return total

def product_list(lst):
    """Return the product of all elements. Empty list returns 1."""
    p = 1
    for x in lst:
        p *= x
    return p

def all_positive(lst):
    """Check if all elements are strictly positive. Empty list returns True."""
    if not lst:
        return True
    for x in lst:
        if x <= 0:
            return False
    return True