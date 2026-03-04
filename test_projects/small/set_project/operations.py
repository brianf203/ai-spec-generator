"""Set operations: union, intersection, difference."""
def union_sets(a, b):
    """Return set of all elements in either a or b."""
    result = set()
    for x in a:
        result.add(x)
    for x in b:
        if x not in result:
            result.add(x)
    return result

def intersection_sets(a, b):
    """Return set of elements in both a and b."""
    result = set()
    for x in a:
        if x in b:
            result.add(x)
    return result

def difference_sets(a, b):
    """Return set of elements in a but not in b."""
    result = set()
    for x in a:
        if x not in b:
            result.add(x)
    return result

def symmetric_difference(a, b):
    """Return set of elements in exactly one of a or b."""
    result = set()
    for x in a:
        if x not in b:
            result.add(x)
    for x in b:
        if x not in a:
            result.add(x)
    return result

def is_subset(a, b):
    """Check if every element of a is in b."""
    for x in a:
        if x not in b:
            return False
    return True

def is_superset(a, b):
    """Check if every element of b is in a."""
    return is_subset(b, a)

def is_disjoint(a, b):
    """Check if a and b have no elements in common."""
    for x in a:
        if x in b:
            return False
    return True