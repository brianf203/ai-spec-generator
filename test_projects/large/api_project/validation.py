"""API validation."""
def validate_json(data):
    """Check if data is valid JSON structure (dict or list)."""
    return isinstance(data, (dict, list))

def required_fields(data, fields):
    """Check all required fields are present."""
    for f in fields:
        if f not in data:
            return False
    return True

def optional_fields(data, fields):
    """Check optional fields that are present are valid."""
    for f in fields:
        if f in data and data[f] is None:
            return False
    return True

def validate_types(data, schema):
    """Validate types of present keys against schema."""
    for k, v in schema.items():
        if k in data and not isinstance(data.get(k), v):
            return False
    return True

def validate_range(n, lo, hi):
    """Check n is in [lo, hi] inclusive."""
    return lo <= n <= hi

def validate_length(s, min_len, max_len):
    """Check string length in range."""
    ln = len(s)
    return min_len <= ln <= max_len

def validate_enum(val, choices):
    """Check val is in choices."""
    return val in choices

def validate_regex(s, pattern):
    """Check s matches pattern."""
    if not s:
        return False
    import re
    return bool(re.match(pattern, s))

def validate_email_format(e):
    """Basic email format check."""
    if "@" not in e:
        return False
    parts = e.split("@")
    return "." in parts[-1]

def validate_uuid(s):
    """Check s looks like UUID format."""
    return len(s) == 36 and s.count("-") == 4

def sanitize_string(s, max_len=1000):
    """Truncate string to max_len."""
    s = s or ""
    return s[:max_len]

def validate_pagination(page, per_page):
    """Validate pagination params."""
    return page >= 1 and 1 <= per_page <= 100

def validate_sort_field(field, allowed):
    """Check sort field is allowed."""
    return field in allowed