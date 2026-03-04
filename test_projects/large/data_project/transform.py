"""Data transform."""
def map_values(d, fn):
    """Apply fn to each value, keep keys."""
    result = {}
    for k, v in d.items():
        result[k] = fn(v)
    return result

def filter_keys(d, keys):
    """Keep only keys in the given set."""
    result = {}
    for k in keys:
        if k in d:
            result[k] = d[k]
    return result

def rename_keys(d, mapping):
    """Rename keys according to mapping."""
    result = {}
    for k, v in d.items():
        new_k = mapping.get(k, k)
        result[new_k] = v
    return result
def flatten_dict(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            flat_inner = flatten_dict(v, key + ".")
            for k2, v2 in flat_inner.items():
                out[k2] = v2
        else:
            out[key] = v
    return out
def nest_dict(d, sep="."):
    out = {}
    for k, v in d.items():
        parts = k.split(sep)
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out
def transform_values(d, fn):
    """Apply fn to each value."""
    result = {}
    for k, v in d.items():
        result[k] = fn(v)
    return result

def filter_values(d, pred):
    """Keep only values where pred is True."""
    result = {}
    for k, v in d.items():
        if pred(v):
            result[k] = v
    return result

def deep_merge(a, b):
    """Recursively merge b into a."""
    result = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result

def pick(d, keys):
    """Extract only specified keys."""
    result = {}
    for k in keys:
        if k in d:
            result[k] = d[k]
    return result

def omit(d, keys):
    """Return dict without specified keys."""
    result = {}
    for k, v in d.items():
        if k not in keys:
            result[k] = v
    return result

def invert_dict(d):
    """Swap keys and values. Values must be hashable."""
    result = {}
    for k, v in d.items():
        result[v] = k
    return result