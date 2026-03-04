"""Dictionary transformation operations."""
def map_values(d, fn):
    """Apply fn to each value, return new dict with same keys."""
    result = {}
    for k, v in d.items():
        result[k] = fn(v)
    return result

def filter_by_value(d, pred):
    """Return dict with only entries where pred(value) is True."""
    result = {}
    for k, v in d.items():
        if pred(v):
            result[k] = v
    return result

def rename_key(d, old_k, new_k):
    """Create new dict with old_k renamed to new_k."""
    result = {}
    for k, v in d.items():
        if k == old_k:
            result[new_k] = v
        else:
            result[k] = v
    return result