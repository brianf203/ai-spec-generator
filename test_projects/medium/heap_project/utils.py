"""Heap utilities."""
from heap import heapify

def heap_from_list(lst):
    """Create heap from list. Returns new list."""
    return heapify(list(lst))

def heap_merge(a, b):
    """Merge two heaps. Returns new heap."""
    return heapify(list(a) + list(b))

def heap_nlargest(h, n):
    """Return n largest elements. Destroys heap."""
    result = []
    for _ in range(min(n, len(h))):
        if not h:
            break
        result.append(heap_pop(h))
    return result[::-1]