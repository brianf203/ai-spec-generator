"""Text formatting."""
def char_count(s):
    """Return number of characters in s."""
    return len(s)

def capitalize_sentence(s):
    """Capitalize first character of s."""
    return s.capitalize()

def remove_punctuation(s):
    """Remove non-alphanumeric characters except spaces."""
    result = []
    for c in s:
        if c.isalnum() or c.isspace():
            result.append(c)
    return "".join(result)

def normalize_whitespace(s):
    """Collapse runs of whitespace to single spaces."""
    return " ".join(s.split())

def indent_lines(s, spaces=4):
    """Add leading spaces to each line."""
    lines = s.split("\n")
    result = []
    for line in lines:
        result.append(" " * spaces + line)
    return "\n".join(result)

def dedent_lines(s):
    """Remove leading whitespace from each line."""
    lines = s.split("\n")
    result = []
    for line in lines:
        result.append(line.lstrip())
    return "\n".join(result)
def wrap_text(s, width):
    words, line, out = s.split(), [], []
    for w in words:
        if line and len(" ".join(line)) + len(w) + 1 > width:
            out.append(" ".join(line))
            line = []
        line.append(w)
    if line:
        out.append(" ".join(line))
    return "\n".join(out)
def truncate_at_word(s, max_len):
    if len(s) <= max_len:
        return s
    truncated = s[:max_len]
    parts = truncated.rsplit(" ", 1)
    return parts[0] + "..."