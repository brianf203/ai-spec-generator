"""Singly linked list node."""
def create_node(value, next=None):
    """Create node with value and optional next."""
    return {"value": value, "next": next}

def get_value(node):
    """Get value from node."""
    return node["value"]

def get_next(node):
    """Get next node."""
    return node.get("next")

def set_next(node, nxt):
    """Set next node."""
    node["next"] = nxt
    return node