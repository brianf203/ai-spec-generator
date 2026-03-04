"""Substring operations."""
def find_substring(s, sub):
    """Find first index of sub in s. -1 if not found."""
    for i in range(len(s) - len(sub) + 1):
        if s[i:i+len(sub)] == sub:
            return i
    return -1

def count_substring(s, sub):
    """Count occurrences of sub in s."""
    count = 0
    i = 0
    while True:
        pos = find_substring(s[i:], sub)
        if pos < 0:
            break
        count += 1
        i += pos + 1
    return count

def replace_first(s, old, new):
    """Replace first occurrence of old with new."""
    pos = find_substring(s, old)
    if pos < 0:
        return s
    return s[:pos] + new + s[pos+len(old):]

def replace_all(s, old, new):
    """Replace all occurrences of old with new."""
    return s.replace(old, new)