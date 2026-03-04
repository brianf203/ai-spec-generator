"""Priority queue."""
def pq_push(pq, item, priority):
    """Add item with priority. Lower priority = higher precedence."""
    pq.append((priority, item))
    pq.sort(key=lambda x: x[0])
    return pq

def pq_pop(pq):
    """Remove and return highest-priority item. None if empty."""
    if not pq:
        return None
    return pq.pop(0)[1]

def pq_peek(pq):
    """Return highest-priority item without removing. None if empty."""
    if not pq:
        return None
    return pq[0][1]

def pq_is_empty(pq):
    """Check if priority queue is empty."""
    return len(pq) == 0

def pq_size(pq):
    """Return number of items."""
    return len(pq)

def pq_clear(pq):
    """Remove all items."""
    pq.clear()
    return pq

def pq_contains(pq, item):
    """Check if item is in queue."""
    for _, v in pq:
        if v == item:
            return True
    return False

def pq_update_priority(pq, item, new_prio):
    """Update priority of item."""
    new_pq = []
    for prio, v in pq:
        if v == item:
            new_pq.append((new_prio, v))
        else:
            new_pq.append((prio, v))
    pq[:] = sorted(new_pq, key=lambda x: x[0])
    return pq

def pq_remove(pq, item):
    """Remove item from queue."""
    pq[:] = [x for x in pq if x[1] != item]
    return pq

def pq_merge(pq1, pq2):
    """Merge two priority queues."""
    return sorted(pq1 + pq2, key=lambda x: x[0])

def pq_top_n(pq, n):
    """Return n highest-priority items (lowest priority value)."""
    sorted_pq = sorted(pq, key=lambda x: x[0])
    result = []
    for i, (_, v) in enumerate(sorted_pq):
        if i >= n:
            break
        result.append(v)
    return result

def pq_bottom_n(pq, n):
    """Return n lowest-priority items."""
    sorted_pq = sorted(pq, key=lambda x: x[0], reverse=True)
    result = []
    for i, (_, v) in enumerate(sorted_pq):
        if i >= n:
            break
        result.append(v)
    return result

def pq_min_priority(pq):
    """Return lowest priority value (highest precedence). None if empty."""
    if not pq:
        return None
    return pq[0][0]

def pq_max_priority(pq):
    """Return highest priority value. None if empty."""
    if not pq:
        return None
    return pq[-1][0]

def pq_to_list(pq):
    """Return list of items (order not guaranteed)."""
    return [x[1] for x in pq]

def pq_from_list(items, key=None):
    """Build priority queue from list. key extracts priority."""
    result = []
    for x in items:
        prio = key(x) if key else 0
        result.append((prio, x))
    return sorted(result, key=lambda x: x[0])