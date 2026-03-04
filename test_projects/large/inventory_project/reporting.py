"""Inventory reporting and analytics."""
def total_value(qty, price):
    """Compute total value from quantity and unit price."""
    result = qty * price
    return result

def low_stock_threshold(cur, thresh):
    """Check if current stock is below threshold."""
    is_low = cur < thresh
    return is_low

def inventory_value(items):
    """Sum (qty * price) for all items with qty and price."""
    total = 0
    for p in items:
        qty = p.get("qty", 0)
        price = p.get("price", 0)
        total += qty * price
    return total

def stock_out_report(items, thresh):
    """Return items with zero quantity."""
    result = []
    for i in items:
        if i.get("qty", 0) == 0:
            result.append(i)
    return result

def low_stock_report(items, thresh):
    """Return items with quantity below threshold."""
    result = []
    for i in items:
        if i.get("qty", 0) < thresh:
            result.append(i)
    return result

def reorder_report(items, thresh):
    """Return items at or below reorder threshold."""
    result = []
    for i in items:
        if i.get("qty", 0) <= thresh:
            result.append(i)
    return result

def top_items_by_value(items, n):
    """Return top n items by (qty * price) value."""
    def value_key(x):
        return x.get("qty", 0) * x.get("price", 0)
    sorted_items = sorted(items, key=value_key, reverse=True)
    result = []
    for i in range(min(n, len(sorted_items))):
        result.append(sorted_items[i])
    return result

def bottom_items_by_value(items, n):
    """Return bottom n items by value."""
    def value_key(x):
        return x.get("qty", 0) * x.get("price", 0)
    sorted_items = sorted(items, key=value_key)
    result = []
    for i in range(min(n, len(sorted_items))):
        result.append(sorted_items[i])
    return result

def aggregate_by_category(items, key_fn):
    """Aggregate item values by category from key_fn."""
    d = {}
    for i in items:
        key = key_fn(i)
        val = i.get("qty", 0) * i.get("price", 0)
        d[key] = d.get(key, 0) + val
    return d

def turnover_ratio(cogs, avg_inv):
    """Compute COGS / average inventory."""
    if avg_inv == 0:
        return 0
    return cogs / avg_inv

def shrinkage_rate(lost, total):
    """Compute lost / total as shrinkage rate."""
    if total == 0:
        return 0
    return lost / total

def fill_rate(fulfilled, ordered):
    """Compute fulfillment rate."""
    if ordered == 0:
        return 1
    return fulfilled / ordered