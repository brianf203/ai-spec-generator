"""Prefix and suffix operations."""
def is_prefix(s, prefix):
    """Check if prefix is prefix of s."""
    if len(prefix) > len(s):
        return False
    return s[:len(prefix)] == prefix

def is_suffix(s, suffix):
    """Check if suffix is suffix of s."""
    if len(suffix) > len(s):
        return False
    return s[-len(suffix):] == suffix

def remove_prefix(s, prefix):
    """Remove prefix if present."""
    if is_prefix(s, prefix):
        return s[len(prefix):]
    return s

def remove_suffix(s, suffix):
    """Remove suffix if present."""
    if is_suffix(s, suffix):
        return s[:-len(suffix)]
    return s

def common_prefix(a, b):
    """Longest common prefix of a and b."""
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return a[:i]

def longest_prefix_match(s, prefixes):
    """Return longest matching prefix from list."""
    best = ""
    for p in prefixes:
        if is_prefix(s, p) and len(p) > len(best):
            best = p
    return best