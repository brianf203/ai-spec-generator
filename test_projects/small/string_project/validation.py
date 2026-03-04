"""String validation helpers."""
def is_empty(s):
    """Check if the string has zero length."""
    return len(s) == 0

def has_digit(s):
    """Check if the string contains at least one digit."""
    for c in s:
        if c.isdigit():
            return True
    return False

def has_alpha(s):
    """Check if the string contains at least one alphabetic character."""
    for c in s:
        if c.isalpha():
            return True
    return False

def starts_with(s, prefix):
    """Check if s begins with the given prefix."""
    return s.startswith(prefix)

def ends_with(s, suffix):
    """Check if s ends with the given suffix."""
    return s.endswith(suffix)