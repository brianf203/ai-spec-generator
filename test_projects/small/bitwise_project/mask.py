"""Bit masking utilities."""
def set_bit(n, pos):
    """Set bit at position pos to 1."""
    return n | (1 << pos)

def clear_bit(n, pos):
    """Clear bit at position pos to 0."""
    return n & ~(1 << pos)

def toggle_bit(n, pos):
    """Toggle bit at position pos."""
    return n ^ (1 << pos)

def get_bit(n, pos):
    """Get bit at position pos (0 or 1)."""
    return (n >> pos) & 1

def mask_lower(n, bits):
    """Keep only lower bits bits."""
    mask = (1 << bits) - 1
    return n & mask