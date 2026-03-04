"""Linked list operations."""
def length(head):
    """Count nodes in list."""
    count = 0
    while head:
        count += 1
        head = head.get("next")
    return count

def to_list(head):
    """Convert to Python list."""
    result = []
    while head:
        result.append(head["value"])
        head = head.get("next")
    return result

def from_list(lst):
    """Create linked list from Python list."""
    head = None
    for x in reversed(lst):
        head = {"value": x, "next": head}
    return head

def append(head, x):
    """Append x. Returns new head if was empty."""
    new = {"value": x, "next": None}
    if not head:
        return new
    cur = head
    while cur.get("next"):
        cur = cur["next"]
    cur["next"] = new
    return head

def reverse(head):
    """Reverse list. Returns new head."""
    prev = None
    while head:
        nxt = head.get("next")
        head["next"] = prev
        prev = head
        head = nxt
    return prev