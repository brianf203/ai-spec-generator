"""Counter metrics."""
def increment(counter, key, amt=1):
    """Increment counter for key."""
    counter[key] = counter.get(key, 0) + amt
    return counter

def get_count(counter, key):
    """Get count for key."""
    return counter.get(key, 0)

def reset_counter(counter, key=None):
    """Reset counter for key or all."""
    if key is None:
        counter.clear()
    else:
        counter[key] = 0
    return counter

def counter_keys(counter):
    """Return all counter keys."""
    return list(counter.keys())