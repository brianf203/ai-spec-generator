"""String utility functions."""
def count_chars(s):
    """Return the number of characters in the string."""
    count = 0
    for _ in s:
        count += 1
    return count

def strip_whitespace(s):
    """Remove leading and trailing whitespace."""
    result = s.strip()
    return result

def remove_spaces(s):
    """Remove all space characters from the string."""
    result = []
    for c in s:
        if c != " ":
            result.append(c)
    return "".join(result)

def first_char(s):
    """Return the first character, or empty string if s is empty."""
    if not s:
        return ""
    return s[0]

def last_char(s):
    """Return the last character, or empty string if s is empty."""
    if not s:
        return ""
    return s[-1]