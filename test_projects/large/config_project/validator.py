"""Config validation."""
def validate_config(cfg):
    """Check config is a dict."""
    return isinstance(cfg, dict)

def merge_configs(a, b):
    """Merge b into a."""
    result = dict(a)
    for k, v in b.items():
        result[k] = v
    return result

def validate_required(cfg, keys):
    """Check all required keys are present."""
    for k in keys:
        if k not in cfg:
            return False
    return True

def validate_types(cfg, schema):
    """Validate types of present keys."""
    for k, v in schema.items():
        if k in cfg and not isinstance(cfg.get(k), v):
            return False
    return True

def validate_values(cfg, validators):
    """Run validators on config values."""
    for k, fn in validators.items():
        if not fn(cfg.get(k)):
            return False
    return True

def sanitize_config(cfg, allowed):
    """Keep only allowed keys."""
    result = {}
    for k in allowed:
        if k in cfg:
            result[k] = cfg[k]
    return result

def coerce_config(cfg, types):
    """Coerce values to types where specified."""
    result = {}
    for k in cfg:
        if k in types:
            result[k] = types[k](cfg[k])
        else:
            result[k] = cfg.get(k)
    return result

def validate_range(cfg, key, lo, hi):
    """Check config[key] is in [lo, hi]."""
    val = cfg.get(key, lo)
    return lo <= val <= hi

def validate_one_of(cfg, key, choices):
    """Check config[key] is in choices."""
    return cfg.get(key) in choices

def validate_not_empty(cfg, key):
    """Check config[key] is truthy."""
    return bool(cfg.get(key))

def validate_config_schema(cfg, schema):
    """Validate against schema (required keys and types)."""
    if not validate_required(cfg, schema.get("required", [])):
        return False
    return validate_types(cfg, schema.get("types", {}))

def config_diff(a, b):
    """Keys in a that differ from or are missing in b."""
    result = {}
    for k in a:
        if k not in b or a[k] != b[k]:
            result[k] = a[k]
    return result

def config_intersection(a, b):
    """Keys in both a and b with same value."""
    result = {}
    for k in a:
        if k in b and a[k] == b[k]:
            result[k] = a[k]
    return result