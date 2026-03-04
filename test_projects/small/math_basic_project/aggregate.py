"""Basic math aggregation."""
def sum_list(lst):
    """Sum all numbers in list."""
    total = 0
    for x in lst:
        total += x
    return total

def product_list(lst):
    """Product of all numbers. Returns 1 for empty."""
    result = 1
    for x in lst:
        result *= x
    return result

def mean_list(lst):
    """Arithmetic mean. Returns 0 for empty."""
    if not lst:
        return 0
    return sum_list(lst) / len(lst)

def min_list(lst):
    """Minimum value. Raises ValueError if empty."""
    if not lst:
        raise ValueError("Empty list")
    m = lst[0]
    for x in lst[1:]:
        if x < m:
            m = x
    return m

def max_list(lst):
    """Maximum value. Raises ValueError if empty."""
    if not lst:
        raise ValueError("Empty list")
    m = lst[0]
    for x in lst[1:]:
        if x > m:
            m = x
    return m