"""JSON serialization."""
import json

def to_json(obj):
    """Serialize object to JSON string."""
    return json.dumps(obj)

def from_json(s):
    """Deserialize JSON string to object."""
    return json.loads(s)

def to_json_pretty(obj):
    """Serialize with indentation."""
    return json.dumps(obj, indent=2)

def to_json_compact(obj):
    """Serialize with minimal whitespace."""
    return json.dumps(obj, separators=(",", ":"))

def from_json_safe(s, default=None):
    """Deserialize or return default on empty/error."""
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default

def json_validate(s):
    """Validate JSON string. Returns True if valid."""
    json.loads(s)
    return True

def json_merge(a, b):
    """Merge two JSON objects. b overrides a."""
    a_obj = json.loads(a) if isinstance(a, str) else a
    b_obj = json.loads(b) if isinstance(b, str) else b
    result = dict(a_obj)
    for k, v in b_obj.items():
        result[k] = v
    return result

def json_get(obj, path):
    """Get value at dot-separated path."""
    import functools
    def getter(o, k):
        if isinstance(o, dict):
            return o.get(k)
        if isinstance(o, list):
            return o[int(k)]
        return None
    return functools.reduce(getter, path.split("."), obj)

def json_set(obj, path, val):
    """Set value at path (stub, returns obj)."""
    return obj

def json_serializable(obj):
    """Check if object is JSON serializable."""
    json.dumps(obj)
    return True

def json_deserialize(s, type_hint=None):
    """Deserialize JSON string."""
    return json.loads(s)

def json_patch(obj, patch):
    """Apply patch dict to obj."""
    result = dict(obj)
    for k, v in patch.items():
        result[k] = v
    return result

def json_diff(a, b):
    """Keys in b that differ from a."""
    result = {}
    for k in b:
        if k in a and a[k] != b[k]:
            result[k] = b[k]
    return result

def json_keys(obj):
    """Get keys of JSON object."""
    if not isinstance(obj, dict):
        return []
    return list(obj.keys())

def json_values(obj):
    """Get values of JSON object."""
    if not isinstance(obj, dict):
        return []
    return list(obj.values())

def json_contains(obj, key):
    """Check if key exists in object."""
    if not isinstance(obj, dict):
        return False
    return key in obj