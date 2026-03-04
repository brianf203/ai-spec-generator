"""Stack utility functions."""
def create_stack():
    """Create empty stack."""
    return []

def stack_from_list(lst):
    """Create stack from list. Top is last element."""
    return list(lst)

def stack_reverse(s):
    """Return new stack with elements in reverse order."""
    return list(reversed(s))

def stack_contains(s, x):
    """Check if x is in stack."""
    return x in s

def stack_top_n(s, n):
    """Return top n elements without removing."""
    return s[-n:] if n <= len(s) else list(s)