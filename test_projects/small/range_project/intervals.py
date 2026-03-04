"""Interval utilities."""
def to_list(r):
    """Convert range to list."""
    return list(r)

def step_range(start, stop, step):
    """Create range with step. Returns list."""
    result = []
    i = start
    while (step > 0 and i < stop) or (step < 0 and i > stop):
        result.append(i)
        i += step
    return result

def chunk_range(n, chunk_size):
    """Split range(n) into chunks of chunk_size."""
    result = []
    for i in range(0, n, chunk_size):
        result.append(list(range(i, min(i + chunk_size, n))))
    return result

def range_sum(r):
    """Sum of all integers in range."""
    return sum(r)

def range_min_max(r):
    """Return (min, max) of range."""
    if not r:
        return (None, None)
    return (min(r), max(r))