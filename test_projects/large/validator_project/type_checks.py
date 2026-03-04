"""Type checking and coercion utilities."""
def is_int(x):
    """Check if x is int type."""
    return isinstance(x, int)

def is_str(x):
    """Check if x is str type."""
    return isinstance(x, str)

def is_list(x):
    """Check if x is list type."""
    return isinstance(x, list)

def is_dict(x):
    """Check if x is dict type."""
    return isinstance(x, dict)

def is_float(x):
    """Check if x is float type."""
    return isinstance(x, float)

def is_bool(x):
    """Check if x is bool type."""
    return isinstance(x, bool)

def is_none(x):
    """Check if x is None."""
    return x is None

def is_number(x):
    """Check if x is int or float."""
    return isinstance(x, (int, float))

def is_sequence(x):
    """Check if x is list or tuple."""
    return isinstance(x, (list, tuple))

def is_mapping(x):
    """Check if x is dict (mapping type)."""
    return isinstance(x, dict)

def coerce_int(x):
    """Convert x to int. Raises on failure."""
    return int(x)

def coerce_str(x):
    """Convert x to str."""
    return str(x)

def coerce_float(x):
    """Convert x to float."""
    return float(x)

def coerce_bool(x):
    """Convert x to bool."""
    return bool(x)

def safe_int(x, default=0):
    """Convert to int, or return default if x is None."""
    if x is None:
        return default
    return int(x)

def safe_str(x, default=""):
    """Convert to str, or return default if x is None."""
    if x is None:
        return default
    return str(x)