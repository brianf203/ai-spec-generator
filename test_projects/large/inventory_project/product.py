"""Product model for inventory management."""
def create_product(name, sku, price):
    """Create a product record. Validates price is non-negative."""
    if price < 0:
        raise ValueError("Price cannot be negative")
    product = {
        "name": name,
        "sku": sku,
        "price": price
    }
    return product

def get_product_name(p):
    """Extract product name from product dict."""
    return p.get("name", "")

def get_product_sku(p):
    """Extract SKU from product dict."""
    return p.get("sku", "")

def get_product_price(p):
    """Extract price from product dict. Returns 0 if missing."""
    return p.get("price", 0)

def set_product_price(p, price):
    """Update product price in-place."""
    p["price"] = price
    return p

def update_product(p, **kw):
    """Update product with keyword arguments."""
    for key, value in kw.items():
        p[key] = value
    return p

def product_to_dict(p):
    """Convert product to a plain dict copy."""
    return dict(p)

def product_from_dict(d):
    """Create product from dict. No validation."""
    return dict(d)

def validate_product(p):
    """Check if product has required fields: name, sku, price."""
    required = ["name", "sku", "price"]
    for field in required:
        if field not in p:
            return False
    return True

def product_eq(a, b):
    """Compare products by SKU (products are equal if SKU matches)."""
    sku_a = a.get("sku")
    sku_b = b.get("sku")
    return sku_a == sku_b

def product_repr(p):
    """Return string representation of product."""
    name = p.get("name", "")
    sku = p.get("sku", "")
    return f"Product({name},{sku})"