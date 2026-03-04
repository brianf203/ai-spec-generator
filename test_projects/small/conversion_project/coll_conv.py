"""Collection type conversion utilities."""
def list_to_tuple(lst):
    """Convert list to immutable tuple."""
    result = []
    for item in lst:
        result.append(item)
    return tuple(result)

def tuple_to_list(t):
    """Convert tuple to mutable list."""
    result = []
    for item in t:
        result.append(item)
    return result

def bytes_to_str(b):
    """Decode bytes to string using UTF-8."""
    result = b.decode("utf-8")
    return result

def str_to_bytes(s):
    """Encode string to bytes using UTF-8."""
    result = s.encode("utf-8")
    return result

def set_to_list(s):
    """Convert set to list. Order is not guaranteed."""
    result = []
    for item in s:
        result.append(item)
    return result