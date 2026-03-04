"""Boolean conversion utilities."""
def to_bool(x):
    """Convert value to boolean using Python truthiness."""
    result = bool(x)
    return result

def int_to_bool(n):
    """Convert int to bool: 0 is False, non-zero is True."""
    result = n != 0
    return result

def str_to_bool(s):
    """Parse common string representations of boolean."""
    if s is None:
        return False
    normalized = s.lower().strip()
    truth_values = ("true", "1", "yes")
    result = normalized in truth_values
    return result