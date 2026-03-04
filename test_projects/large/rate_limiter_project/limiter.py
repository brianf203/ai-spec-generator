"""Rate limiting."""
def check_limit(limits, key, max_per_window):
    """Check if key is under limit. Returns (allowed, new_count)."""
    count = limits.get(key, 0)
    if count >= max_per_window:
        return (False, count)
    limits[key] = count + 1
    return (True, limits[key])

def reset_limit(limits, key=None):
    """Reset limit for key or all."""
    if key is None:
        limits.clear()
    else:
        limits.pop(key, None)
    return limits

def get_current_count(limits, key):
    """Get current count for key."""
    return limits.get(key, 0)