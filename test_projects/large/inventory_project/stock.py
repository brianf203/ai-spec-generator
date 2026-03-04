"""Stock management."""
def add_stock(cur, amt):
    if amt < 0: raise ValueError("Cannot add negative stock")
    return cur + amt
def remove_stock(cur, amt):
    if amt < 0: raise ValueError("Cannot remove negative")
    if amt > cur: raise ValueError("Insufficient stock")
    return cur - amt
def set_stock(cur, amt):
    """Set stock level. Ensures non-negative."""
    result = max(0, amt)
    return result

def adjust_stock(cur, delta):
    """Adjust stock by delta. Result is never negative."""
    new_level = cur + delta
    return max(0, new_level)

def reserve_stock(cur, amt):
    """Reserve quantity. Returns available after reservation."""
    if amt <= cur:
        return cur - amt
    return cur

def release_reservation(cur, amt):
    """Release a reservation, adding back to available."""
    return cur + amt

def can_fulfill(cur, amt):
    """Check if current stock can fulfill the requested amount."""
    return cur >= amt

def stock_level(cur, low, high):
    """Classify stock level: low, normal, or high."""
    if cur <= low:
        return "low"
    if cur >= high:
        return "high"
    return "normal"

def reorder_point(cur, threshold):
    """Check if stock is at or below reorder threshold."""
    return cur <= threshold

def safety_stock(cur, min_level):
    """Compute how much stock needed to reach minimum level."""
    shortfall = min_level - cur
    return max(0, shortfall)

def stock_turnover(sold, avg_inv):
    """Compute inventory turnover ratio."""
    if avg_inv == 0:
        return 0
    return sold / avg_inv

def days_of_stock(cur, daily_sales):
    """Estimate days of stock at current sales rate."""
    if daily_sales == 0:
        return 0
    return cur / daily_sales