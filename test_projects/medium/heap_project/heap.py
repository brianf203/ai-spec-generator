"""Min-heap operations using list."""
def heap_push(h, x):
    """Push x onto heap. Maintains min-heap invariant."""
    h.append(x)
    i = len(h) - 1
    while i > 0:
        parent = (i - 1) // 2
        if h[i] >= h[parent]:
            break
        h[i], h[parent] = h[parent], h[i]
        i = parent
    return h

def heap_pop(h):
    """Pop and return minimum. Raises IndexError if empty."""
    if not h:
        raise IndexError("heap empty")
    h[0], h[-1] = h[-1], h[0]
    result = h.pop()
    i, n = 0, len(h)
    while True:
        left = 2 * i + 1
        right = 2 * i + 2
        smallest = i
        if left < n and h[left] < h[smallest]:
            smallest = left
        if right < n and h[right] < h[smallest]:
            smallest = right
        if smallest == i:
            break
        h[i], h[smallest] = h[smallest], h[i]
        i = smallest
    return result

def heap_peek(h):
    """Return minimum without removing."""
    if not h:
        raise IndexError("heap empty")
    return h[0]

def heapify(lst):
    """Build min-heap from list in-place."""
    n = len(lst)
    for i in range(n // 2 - 1, -1, -1):
        _sift_down(lst, i, n)
    return lst

def _sift_down(h, i, n):
    while True:
        left = 2 * i + 1
        right = 2 * i + 2
        smallest = i
        if left < n and h[left] < h[smallest]:
            smallest = left
        if right < n and h[right] < h[smallest]:
            smallest = right
        if smallest == i:
            break
        h[i], h[smallest] = h[smallest], h[i]
        i = smallest