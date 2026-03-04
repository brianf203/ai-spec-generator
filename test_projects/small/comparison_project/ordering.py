"""Ordering helpers."""
def compare_asc(a, b):
    """Return -1 if a<b, 0 if a==b, 1 if a>b."""
    if a < b:
        return -1
    if a > b:
        return 1
    return 0

def compare_desc(a, b):
    """Reverse order: 1 if a<b, -1 if a>b."""
    return -compare_asc(a, b)

def rank_in_list(lst, x):
    """Return 0-based rank of x in sorted list."""
    sorted_vals = sorted(lst)
    for i, v in enumerate(sorted_vals):
        if v == x:
            return i
    return -1

def percentile_rank(lst, x):
    """Return percentile rank of x (0-100)."""
    if not lst:
        return 0
    below = sum(1 for v in lst if v < x)
    return 100 * below / len(lst)