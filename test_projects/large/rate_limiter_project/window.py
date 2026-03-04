"""Time window helpers."""
def window_key(ts, window_sec):
    """Bucket ts into window."""
    return ts // window_sec

def is_in_window(ts, start, window_sec):
    """Check if ts in [start, start+window_sec)."""
    return start <= ts < start + window_sec

def next_window_start(ts, window_sec):
    """Start of next window after ts."""
    return ((ts // window_sec) + 1) * window_sec