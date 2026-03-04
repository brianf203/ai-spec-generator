"""String padding, truncation and number formatting."""
def pad_string(s, width, char=" "):
    """Pad string on right to width, truncate if too long."""
    current_len = len(s)
    if current_len >= width:
        result = s[:width]
        return result
    pad_count = width - current_len
    padded = s + (char * pad_count)
    return padded[:width]

def truncate_string(s, max_len):
    """Truncate string to max_len if longer, else return as-is."""
    if len(s) <= max_len:
        return s
    result = s[:max_len]
    return result

def format_number(n, decimals=2):
    """Format number with fixed decimal places."""
    fmt = "{:." + str(decimals) + "f}"
    result = fmt.format(n)
    return result

def pad_left(s, width, char=" "):
    """Pad string on left to reach width."""
    if len(s) >= width:
        return s
    pad_count = width - len(s)
    padding = char * pad_count
    return padding + s

def center_string(s, width, char=" "):
    """Center string within width using padding."""
    if len(s) >= width:
        return s[:width]
    total_pad = width - len(s)
    left = total_pad // 2
    right = total_pad - left
    result = (char * left) + s + (char * right)
    return result