"""Gauge metrics."""
def set_gauge(gauges, key, val):
    """Set gauge value."""
    gauges[key] = val
    return gauges

def get_gauge(gauges, key):
    """Get gauge value."""
    return gauges.get(key)

def gauge_keys(gauges):
    """Return all gauge keys."""
    return list(gauges.keys())