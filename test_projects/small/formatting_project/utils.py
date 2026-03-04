"""Formatting utilities for percentages, currency, etc."""
def format_percentage(v, decimals=1):
    """Format value as percentage string."""
    fmt = "{:." + str(decimals) + "f}%"
    result = fmt.format(v)
    return result

def remove_extra_spaces(s):
    """Collapse multiple spaces to single space, trim ends."""
    parts = s.split()
    result = " ".join(parts)
    return result

def format_currency(v):
    """Format number as US currency."""
    return "${:.2f}".format(v)

def format_int(n):
    """Format integer with thousands separators."""
    s = str(n)
    if n < 0:
        s = s[1:]
        neg = True
    else:
        neg = False
    result = []
    for i, c in enumerate(reversed(s)):
        if i > 0 and i % 3 == 0:
            result.append(",")
        result.append(c)
    out = "".join(reversed(result))
    return "-" + out if neg else out

def ellipsis(s, n):
    """Truncate to n chars and append ... if truncated."""
    if len(s) <= n:
        return s
    truncated = s[:n]
    return truncated + "..."