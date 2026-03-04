"""Session management."""
def create_session(store, user_id):
    """Create new session. Returns session id."""
    import uuid
    sid = str(uuid.uuid4())
    store[sid] = {"user_id": user_id, "created": 0}
    return sid

def get_session(store, sid):
    """Get session by id."""
    return store.get(sid)

def destroy_session(store, sid):
    """Remove session."""
    if sid in store:
        del store[sid]
    return store

def session_user(store, sid):
    """Get user_id from session."""
    s = store.get(sid)
    return s["user_id"] if s else None

def list_sessions(store):
    """Return all session ids."""
    return list(store.keys())