"""Pickle serialization."""
import pickle

def to_pickle(obj):
    """Serialize object to pickle bytes."""
    return pickle.dumps(obj)

def from_pickle(b):
    """Deserialize from pickle bytes."""
    return pickle.loads(b)

def to_pickle_protocol(obj, protocol=4):
    """Serialize with specific protocol."""
    return pickle.dumps(obj, protocol=protocol)

def from_pickle_safe(b, default=None):
    """Deserialize or return default on empty/error."""
    if not b:
        return default
    try:
        return pickle.loads(b)
    except Exception:
        return default

def pickle_copy(obj):
    """Deep copy via pickle roundtrip."""
    return pickle.loads(pickle.dumps(obj))

def pickle_size(obj):
    """Return size of serialized object in bytes."""
    return len(pickle.dumps(obj))

def pickle_compare(a, b):
    """Compare objects by pickle serialization."""
    return pickle.dumps(a) == pickle.dumps(b)

def to_base64_pickle(obj):
    """Serialize to base64 string."""
    import base64
    data = pickle.dumps(obj)
    return base64.b64encode(data).decode()

def from_base64_pickle(s):
    """Deserialize from base64 string."""
    import base64
    data = base64.b64decode(s)
    return pickle.loads(data)

def pickle_serializable(obj):
    """Check if object is pickle serializable."""
    pickle.dumps(obj)
    return True

def pickle_version(obj):
    """Get format version of pickle."""
    return pickle.format_version

def pickle_high_protocol():
    """Return highest protocol version."""
    return 4

def pickle_compat_load(b):
    """Load pickle compatibly."""
    return pickle.loads(b)

def pickle_dump_to_file(obj, path):
    """Write object to file."""
    with open(path, "wb") as f:
        f.write(pickle.dumps(obj))

def pickle_load_from_file(path):
    """Load object from file."""
    with open(path, "rb") as f:
        return pickle.loads(f.read())

def pickle_roundtrip(obj):
    """Serialize and deserialize."""
    return from_pickle(to_pickle(obj))