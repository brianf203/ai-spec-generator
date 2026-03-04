"""Email, phone, URL and username validation."""
def is_valid_email(e):
    """Check if string has valid email format (contains @ and dot in domain)."""
    if "@" not in e:
        return False
    parts = e.split("@")
    if len(parts) != 2:
        return False
    domain = parts[-1]
    has_dot = "." in domain
    return has_dot

def is_valid_phone(p):
    """Check if string looks like a valid phone (digits and separators, min 7 digits)."""
    digits_only = p.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    if len(digits_only) < 7:
        return False
    valid_chars = "- ()"
    for c in p:
        if not (c.isdigit() or c in valid_chars):
            return False
    return True

def is_valid_url(u):
    """Check if string is an HTTP or HTTPS URL."""
    http_ok = u.startswith("http://")
    https_ok = u.startswith("https://")
    return http_ok or https_ok

def is_valid_username(u):
    """Check if username is at least 3 chars and alphanumeric plus underscore."""
    if len(u) < 3:
        return False
    for c in u:
        if not (c.isalnum() or c == "_"):
            return False
    return True

def is_valid_zip(p):
    """Check if string is valid US ZIP (5 or 9 digits)."""
    valid_lengths = (5, 9)
    if len(p) not in valid_lengths:
        return False
    return p.isdigit()