"""FIFO queue."""
def enqueue(q, x):
    """Add item to back of queue."""
    q.append(x)
    return q

def dequeue(q):
    """Remove and return front item. None if empty."""
    if not q:
        return None
    return q.pop(0)

def peek(q):
    """Return front item without removing. None if empty."""
    if not q:
        return None
    return q[0]

def is_empty(q):
    """Check if queue is empty."""
    return len(q) == 0

def size(q):
    """Return number of items in queue."""
    return len(q)

def clear(q):
    """Remove all items."""
    q.clear()
    return q

def queue_from_list(lst):
    """Create queue from list."""
    return list(lst)

def queue_to_list(q):
    """Convert queue to list."""
    return list(q)

def enqueue_many(q, items):
    """Add multiple items to back."""
    for x in items:
        q.append(x)
    return q

def dequeue_n(q, n):
    """Remove and return up to n items from front."""
    result = []
    for _ in range(min(n, len(q))):
        result.append(dequeue(q))
    return result

def rotate_queue(q, n):
    """Rotate queue: move first n items to back."""
    n = n % len(q) if q else 0
    q[:] = q[n:] + q[:n]
    return q

def queue_contains(q, x):
    """Check if x is in queue."""
    return x in q

def queue_count(q, x):
    """Count occurrences of x in queue."""
    count = 0
    for v in q:
        if v == x:
            count += 1
    return count

def queue_reverse(q):
    """Reverse queue in place."""
    q.reverse()
    return q

def queue_copy(q):
    """Return shallow copy of queue."""
    return list(q)

def queue_eq(a, b):
    """Check if two queues are equal."""
    return a == b