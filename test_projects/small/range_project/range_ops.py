"""Range and interval operations."""
def range_len(r):
    """Return length of range (number of integers)."""
    return len(r)

def range_contains(r, x):
    """Check if x is in range."""
    return x in r

def range_overlaps(a, b):
    """Check if ranges a and b overlap."""
    return a.start < b.stop and b.start < a.stop

def range_merge(a, b):
    """Merge two overlapping ranges. Assumes they overlap."""
    start = min(a.start, b.start)
    stop = max(a.stop, b.stop)
    return range(start, stop)

def range_gap(a, b):
    """Return gap between non-overlapping ranges. Returns None if overlap."""
    if range_overlaps(a, b):
        return None
    if a.stop <= b.start:
        return range(a.stop, b.start)
    return range(b.stop, a.start)

def range_span(ranges):
    """Return minimal range covering all."""
    if not ranges:
        return range(0, 0)
    starts = [r.start for r in ranges]
    stops = [r.stop for r in ranges]
    return range(min(starts), max(stops))