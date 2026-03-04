"""Notification queue."""
def enqueue_notification(q, n):
    """Add notification to queue."""
    q.append(n)
    return q

def dequeue_notification(q):
    """Remove and return next. None if empty."""
    if not q:
        return None
    return q.pop(0)

def queue_size(q):
    """Return queue size."""
    return len(q)

def queue_peek(q):
    """Return next without removing."""
    if not q:
        return None
    return q[0]