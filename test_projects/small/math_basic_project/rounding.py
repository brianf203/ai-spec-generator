"""Rounding utilities."""
def round_to_int(x):
    """Round to nearest integer."""
    return round(x)

def round_to_n(x, n):
    """Round x to n decimal places."""
    factor = 10 ** n
    return round(x * factor) / factor

def floor_val(x):
    """Floor of x."""
    import math
    return math.floor(x)

def ceil_val(x):
    """Ceiling of x."""
    import math
    return math.ceil(x)

def trunc_val(x):
    """Truncate toward zero."""
    import math
    return math.trunc(x)