"""Inventory utility functions."""
def format_currency(amt):
    """Format amount as US currency string."""
    return "${:.2f}".format(amt)

def clamp(v, lo, hi):
    """Clamp value to range [lo, hi]."""
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v

def parse_sku(sku):
    """Normalize SKU: strip whitespace and uppercase."""
    return sku.strip().upper()

def validate_sku(sku):
    """Check SKU is at least 3 chars and alphanumeric."""
    if len(sku) < 3:
        return False
    return sku.isalnum()

def round_price(p):
    """Round price to 2 decimal places."""
    return round(p, 2)

def apply_discount(price, pct):
    """Apply percentage discount to price."""
    discounted = price * (1 - pct / 100)
    return round(discounted, 2)

def apply_tax(price, rate):
    """Apply percentage tax to price."""
    taxed = price * (1 + rate / 100)
    return round(taxed, 2)

def margin(cost, price):
    """Compute profit margin: (price - cost) / price."""
    if price == 0:
        return 0
    return (price - cost) / price

def markup(cost, price):
    """Compute markup: (price - cost) / cost."""
    if cost == 0:
        return 0
    return (price - cost) / cost

def barcode_check_digit(bc):
    """Compute UPC check digit from barcode digits."""
    total = 0
    for i, c in enumerate(bc):
        weight = 3 if i % 2 else 1
        total += int(c) * weight
    return (10 - (total % 10)) % 10

def generate_sku(prefix, n):
    """Generate SKU from prefix and zero-padded number."""
    return "{}{:06d}".format(prefix, n)

def safe_divide(a, b, default=0):
    """Divide a by b, return default if b is zero."""
    if b == 0:
        return default
    return a / b