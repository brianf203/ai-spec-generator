"""TTL (time-to-live) cache with expiry tracking."""
def ttl_get(cache, key, now=0):
    """Get value if not expired. Returns None if expired or missing."""
    expiry = cache.get("_expiry", {}).get(key, 0)
    if expiry <= now:
        return None
    return cache.get(key)

def ttl_set(cache, key, val, ttl):
    """Set key with expiry timestamp ttl."""
    cache[key] = val
    if "_expiry" not in cache:
        cache["_expiry"] = {}
    cache["_expiry"][key] = ttl
    return cache

def ttl_delete(cache, key):
    """Remove key and its expiry."""
    if key in cache:
        del cache[key]
    if "_expiry" in cache and key in cache["_expiry"]:
        del cache["_expiry"][key]
    return cache

def ttl_clear(cache):
    """Clear all cache entries."""
    cache.clear()
    return cache

def ttl_has(cache, key, now=0):
    """Check if key exists and is not expired."""
    return ttl_get(cache, key, now) is not None

def ttl_expiry(cache, key):
    """Get expiry timestamp for key."""
    return cache.get("_expiry", {}).get(key, 0)

def ttl_extend(cache, key, extra_ttl):
    """Extend TTL by extra_ttl."""
    expiry = cache.get("_expiry", {})
    current = expiry.get(key, 0)
    expiry[key] = current + extra_ttl
    cache["_expiry"] = expiry
    return cache

def ttl_cleanup(cache, now):
    """Remove expired entries from expiry map."""
    expiry = cache.get("_expiry", {})
    cache["_expiry"] = {k: v for k, v in expiry.items() if v > now}
    return cache

def ttl_keys(cache, now=0):
    """Return list of non-expired keys."""
    expiry = cache.get("_expiry", {})
    return [k for k in cache if not k.startswith("_") and expiry.get(k, 0) > now]

def ttl_size(cache):
    """Return count of user keys."""
    return len([k for k in cache if not k.startswith("_")])

def ttl_is_expired(cache, key, now):
    """Check if key has expired."""
    return cache.get("_expiry", {}).get(key, 0) <= now

def ttl_remaining(cache, key, now):
    """Seconds until key expires."""
    exp = cache.get("_expiry", {}).get(key, 0)
    return max(0, exp - now)

def ttl_bulk_set(cache, items, ttl):
    """Set multiple keys with same TTL."""
    for k, v in items:
        ttl_set(cache, k, v, ttl)
    return cache

def ttl_bulk_get(cache, keys, now=0):
    """Get multiple keys. Returns dict of key -> value for non-expired."""
    result = {}
    for k in keys:
        val = ttl_get(cache, k, now)
        if val is not None:
            result[k] = val
    return result