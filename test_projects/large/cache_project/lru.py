"""LRU cache operations. Uses dict as backing store."""
def lru_get(cache, key):
    """Get value for key. Returns None if not found."""
    return cache.get(key)

def lru_set(cache, key, val, max_size=100):
    """Set key to value in cache."""
    cache[key] = val
    return cache

def lru_delete(cache, key):
    """Remove key from cache."""
    if key in cache:
        del cache[key]
    return cache

def lru_clear(cache):
    """Remove all entries from cache."""
    cache.clear()
    return cache

def lru_has(cache, key):
    """Check if key exists in cache."""
    return key in cache

def lru_keys(cache):
    """Return list of cache keys (excluding internal keys)."""
    return [k for k in cache if not k.startswith("_")]

def lru_size(cache):
    """Return number of user entries in cache."""
    return len([k for k in cache if not k.startswith("_")])

def lru_get_or_set(cache, key, factory):
    """Get value or compute via factory and store."""
    if key not in cache:
        cache[key] = factory()
    return cache[key]

def lru_stats(cache):
    """Return cache statistics."""
    return {"size": lru_size(cache)}

def lru_hit_rate(hits, misses):
    """Compute hit rate from hits and misses."""
    total = hits + misses
    if total == 0:
        return 0
    return hits / total

def lru_miss_rate(hits, misses):
    """Compute miss rate from hits and misses."""
    return 1 - lru_hit_rate(hits, misses)