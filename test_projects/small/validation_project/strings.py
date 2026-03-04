"""String validation utilities."""
def is_non_empty(s):
    """Check if string has at least one character."""
    result = len(s) > 0
    return result

def has_min_length(s, n):
    """Check if string length is at least n."""
    result = len(s) >= n
    return result

def has_max_length(s, n):
    """Check if string length is at most n."""
    result = len(s) <= n
    return result

def is_alphanumeric(s):
    """Check if string contains only alphanumeric characters."""
    if not s:
        return False
    result = s.isalnum()
    return result

def is_numeric_str(s):
    """Check if string represents an integer (optional leading minus)."""
    if not s:
        return False
    if s.startswith("-"):
        rest = s[1:]
        return rest.isdigit()
    return s.isdigit()