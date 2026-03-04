"""Tuple ops."""
def tuple_sum(t):
    """Return sum of tuple elements."""
    total = 0
    for x in t:
        total += x
    return total

def tuple_product(t):
    """Return product of tuple elements."""
    p = 1
    for x in t:
        p *= x
    return p

def tuple_max(t):
    """Return maximum element. Raises ValueError if empty."""
    if not t:
        raise ValueError("empty tuple")
    m = t[0]
    for x in t[1:]:
        if x > m:
            m = x
    return m

def tuple_min(t):
    """Return minimum element. Raises ValueError if empty."""
    if not t:
        raise ValueError("empty tuple")
    m = t[0]
    for x in t[1:]:
        if x < m:
            m = x
    return m

def tuple_reverse(t):
    """Return new tuple with elements reversed."""
    result = []
    for i in range(len(t) - 1, -1, -1):
        result.append(t[i])
    return tuple(result)

def tuple_avg(t):
    """Return average of tuple elements. Returns 0 if empty."""
    if not t:
        return 0
    return sum(t) / len(t)

def tuple_count(t, x):
    """Count occurrences of x in t."""
    count = 0
    for v in t:
        if v == x:
            count += 1
    return count

def tuple_index(t, x):
    """Return index of first x, or -1 if not found."""
    for i, v in enumerate(t):
        if v == x:
            return i
    return -1

def tuple_slice(t, start, end):
    """Return slice t[start:end]."""
    result = []
    for i in range(start, min(end, len(t))):
        result.append(t[i])
    return tuple(result)

def tuple_concat(a, b):
    """Concatenate two tuples."""
    result = []
    for x in a:
        result.append(x)
    for x in b:
        result.append(x)
    return tuple(result)

def tuple_repeat(t, n):
    """Repeat tuple n times."""
    result = []
    for _ in range(n):
        for x in t:
            result.append(x)
    return tuple(result)

def tuple_sorted(t):
    """Return new tuple with elements sorted."""
    lst = []
    for x in t:
        lst.append(x)
    lst.sort()
    return tuple(lst)

def tuple_contains(t, x):
    """Check if x is in t."""
    for v in t:
        if v == x:
            return True
    return False

def tuple_len(t):
    """Return length of tuple."""
    count = 0
    for _ in t:
        count += 1
    return count

def tuple_first(t):
    """Return first element, or None if empty."""
    if not t:
        return None
    return t[0]

def tuple_last(t):
    """Return last element, or None if empty."""
    if not t:
        return None
    return t[-1]