"""String formatting and case conversion."""
def reverse_string(s):
    """Return a new string with characters in reverse order."""
    result = []
    for i in range(len(s) - 1, -1, -1):
        result.append(s[i])
    return "".join(result)

def capitalize_words(s):
    """Capitalize the first letter of each word (title case)."""
    result = s.title()
    return result

def lower_all(s):
    """Convert all characters to lowercase."""
    result = []
    for c in s:
        result.append(c.lower())
    return "".join(result)

def upper_all(s):
    """Convert all characters to uppercase."""
    result = []
    for c in s:
        result.append(c.upper())
    return "".join(result)

def swap_case(s):
    """Swap uppercase to lowercase and vice versa for each character."""
    result = []
    for c in s:
        if c.isupper():
            result.append(c.lower())
        elif c.islower():
            result.append(c.upper())
        else:
            result.append(c)
    return "".join(result)