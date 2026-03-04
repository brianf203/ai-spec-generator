"""Session management helpers."""
def create_session(user_id):
    """Create a new session for user."""
    session = {
        "user_id": user_id,
        "active": True
    }
    return session

def is_session_active(s):
    """Check if session is marked active."""
    return s.get("active", False)

def invalidate_session(s):
    """Mark session as inactive."""
    s["active"] = False
    return s

def get_session_user(s):
    """Get user_id from session."""
    return s.get("user_id")

def session_ttl(s, created_at, ttl_sec):
    """Check if session is still within TTL."""
    import time
    expiry = created_at + ttl_sec
    return time.time() < expiry

def extend_session(s, extra_ttl):
    """Extend session expiry by extra_ttl seconds."""
    current = s.get("expires", 0)
    s["expires"] = current + extra_ttl
    return s

def session_to_cookie(s):
    """Serialize session user_id for cookie."""
    user_id = s.get("user_id", "")
    return str(user_id)

def cookie_to_session(c):
    """Parse cookie value back to session dict."""
    if c and c.isdigit():
        user_id = int(c)
    else:
        user_id = 0
    return {"user_id": user_id, "active": True}

def merge_sessions(s1, s2):
    """Merge two session dicts. s2 overrides s1."""
    result = dict(s1)
    for k, v in s2.items():
        result[k] = v
    return result

def session_metadata(s):
    """Extract non-core session fields (exclude user_id, active)."""
    result = {}
    for k, v in s.items():
        if k not in ("user_id", "active"):
            result[k] = v
    return result

def is_session_expired(s, now):
    """Check if session has expired given current timestamp."""
    expires = s.get("expires", 0)
    return expires < now

def refresh_session(s):
    """Mark session as refreshed."""
    s["refreshed"] = True
    return s