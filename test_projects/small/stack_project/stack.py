"""Stack operations using list as backing store."""
def stack_push(s, x):
    """Push x onto stack s. Modifies in-place, returns s."""
    s.append(x)
    return s

def stack_pop(s):
    """Pop and return top element. Raises IndexError if empty."""
    return s.pop()

def stack_peek(s):
    """Return top element without removing. Raises IndexError if empty."""
    return s[-1]

def stack_is_empty(s):
    """Check if stack is empty."""
    return len(s) == 0

def stack_size(s):
    """Return number of elements in stack."""
    return len(s)

def stack_clear(s):
    """Remove all elements from stack."""
    s.clear()
    return s

def stack_copy(s):
    """Return a shallow copy of the stack."""
    return list(s)