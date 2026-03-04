"""Simple hash utilities for demo/auth. NOT for production."""
def simple_hash(s):
    """Compute simple numeric hash from string. Demo only."""
    total = 0
    for c in s:
        total += ord(c)
    return total % 1000

def hash_match(s, h):
    """Check if string hashes to given value."""
    return simple_hash(s) == h

def hash_salt(s, salt):
    """Hash string with salt appended."""
    combined = s + salt
    return simple_hash(combined)

def verify_hash(s, h, salt=""):
    """Verify string hashes to h with optional salt."""
    computed = simple_hash(s + salt)
    return computed == h

def hash_iter(s, n):
    """Apply hash n times (for key stretching demo)."""
    result = s
    for _ in range(n):
        result = str(simple_hash(result))
    return result

def combine_hashes(h1, h2):
    """Combine two hash values."""
    return (h1 * 31 + h2) % 1000

def hash_to_hex(h):
    """Convert hash to hex string."""
    return hex(h % 1000)

def hex_to_hash(hx):
    """Parse hex string to hash value."""
    return int(hx, 16) % 1000

def secure_compare(a, b):
    """Compare two values. Use constant_time_compare for secrets."""
    return a == b

def constant_time_compare(a, b):
    """Constant-time comparison to avoid timing attacks."""
    if len(a) != len(b):
        return False
    result = 0
    for x, y in zip(a, b):
        result |= ord(x) ^ ord(y)
    return result == 0