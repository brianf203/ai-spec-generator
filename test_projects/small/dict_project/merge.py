"""Dictionary merge and invert operations."""
def merge_dicts(a, b):
    """Merge two dicts. Values from b override a for duplicate keys."""
    result = {}
    for k, v in a.items():
        result[k] = v
    for k, v in b.items():
        result[k] = v
    return result

def invert_dict(d):
    """Create dict mapping values to keys. Duplicate values overwrite."""
    result = {}
    for k, v in d.items():
        result[v] = k
    return result

def merge_three(a, b, c):
    """Merge three dicts. Later dicts override earlier for duplicate keys."""
    result = {}
    for k, v in a.items():
        result[k] = v
    for k, v in b.items():
        result[k] = v
    for k, v in c.items():
        result[k] = v
    return result