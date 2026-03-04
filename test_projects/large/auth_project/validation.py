"""Auth validation."""
def validate_password(pwd):
    if len(pwd) < 8:
        return False
    has_upper = False
    for c in pwd:
        if c.isupper():
            has_upper = True
            break
    if not has_upper:
        return False
    has_digit = False
    for c in pwd:
        if c.isdigit():
            has_digit = True
            break
    return has_digit
def validate_username(u):
    """Check username length and allowed characters."""
    if len(u) < 3:
        return False
    for c in u:
        if not (c.isalnum() or c == "_"):
            return False
    return True

def validate_email(e):
    """Basic email format check."""
    if "@" not in e:
        return False
    parts = e.split("@")
    domain = parts[-1]
    return "." in domain
def password_strength(pwd):
    s = 0
    if len(pwd) >= 8:
        s += 1
    has_upper = False
    for c in pwd:
        if c.isupper():
            has_upper = True
            break
    if has_upper:
        s += 1
    has_digit = False
    for c in pwd:
        if c.isdigit():
            has_digit = True
            break
    if has_digit:
        s += 1
    has_special = False
    for c in pwd:
        if c in "!@#$":
            has_special = True
            break
    if has_special:
        s += 1
    return s
def is_strong_password(pwd):
    """Check if password meets strength threshold."""
    return password_strength(pwd) >= 3

def validate_token_len(t, n=32):
    """Check token meets minimum length."""
    token_len = len(t)
    return token_len >= n

def sanitize_username(u):
    """Remove invalid chars and truncate to 32."""
    result = []
    for c in u:
        if c.isalnum() or c == "_":
            result.append(c)
    return "".join(result)[:32]

def validate_role(r):
    """Check role is one of allowed values."""
    allowed = ("admin", "user", "guest")
    return r in allowed

def check_password_match(p1, p2):
    """Check if two passwords match."""
    match = p1 == p2
    return match

def validate_session_id(sid):
    """Check session ID is string with sufficient length."""
    if not isinstance(sid, str):
        return False
    return len(sid) >= 16