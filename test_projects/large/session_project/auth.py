"""Session auth helpers."""
def is_authenticated(store, sid):
    """Check if session exists."""
    return sid in store

def require_session(store, sid):
    """Return session or raise."""
    if sid not in store:
        raise KeyError("Session not found")
    return store[sid]

def session_count(store):
    """Count active sessions."""
    return len(store)