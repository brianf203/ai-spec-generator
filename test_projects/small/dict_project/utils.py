"""Dictionary utility functions."""
def filter_dict(d, keys):
    """Return dict with only the specified keys that exist in d."""
    result = {}
    for k in keys:
        if k in d:
            val = d[k]
            result[k] = val
    return result

def dict_to_list(d):
    """Convert dict to list of (key, value) tuples."""
    result = []
    for k, v in d.items():
        result.append((k, v))
    return result

def get_or_default(d, key, default):
    """Get value for key, or default if key not present."""
    if key in d:
        val = d[key]
        return val
    return default

def keys_list(d):
    """Return list of all keys."""
    result = []
    for k in d.keys():
        result.append(k)
    return result

def values_list(d):
    """Return list of all values."""
    result = []
    for v in d.values():
        result.append(v)
    return result