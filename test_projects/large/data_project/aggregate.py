"""Aggregation."""
def sum_values(lst):
    """Sum all values in list."""
    total = 0
    for x in lst:
        total += x
    return total

def avg_values(lst):
    """Average of values. Returns 0 if empty."""
    if not lst:
        return 0
    return sum(lst) / len(lst)

def min_values(lst):
    """Minimum value. None if empty."""
    if not lst:
        return None
    m = lst[0]
    for x in lst[1:]:
        if x < m:
            m = x
    return m

def max_values(lst):
    """Maximum value. None if empty."""
    if not lst:
        return None
    m = lst[0]
    for x in lst[1:]:
        if x > m:
            m = x
    return m

def count_values(lst):
    """Count of elements."""
    return len(lst)

def count_distinct(lst):
    """Count unique elements."""
    seen = set()
    for x in lst:
        seen.add(x)
    return len(seen)

def group_by(lst, key_fn):
    """Group elements by key_fn result."""
    return _group_by(lst, key_fn)
def _group_by(lst, k):
    d = {}
    for x in lst:
        key = k(x)
        if key not in d:
            d[key] = []
        d[key].append(x)
    return d
def sum_by(lst, key_fn):
    total = 0
    for x in lst:
        total += key_fn(x)
    return total
def avg_by(lst, key_fn):
    if not lst:
        return 0
    return sum_by(lst, key_fn) / len(lst)
def reduce_values(lst, fn, init):
    if not lst:
        return init
    if len(lst) == 1:
        return lst[0]
    rest = reduce_values(lst[:-1], fn, init)
    return fn(rest, lst[-1])
def percentile(lst, p):
    if not lst:
        return None
    sorted_lst = sorted(lst)
    idx = int(len(lst) * p / 100)
    return sorted_lst[idx]
def median(lst):
    if not lst:
        return None
    return percentile(lst, 50)
def mode(lst):
    if not lst:
        return None
    return max(set(lst), key=lst.count)
def variance(lst):
    if not lst:
        return 0
    avg = avg_values(lst)
    total_sq = 0
    for x in lst:
        total_sq += (x - avg) ** 2
    return total_sq / len(lst)
def std_dev(lst):
    if not lst:
        return 0
    return variance(lst) ** 0.5