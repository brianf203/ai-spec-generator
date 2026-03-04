"""Config loader."""
def load_config(path):
    """Load config from path. Returns minimal dict with path key."""
    return {"path": path}

def get_config_value(cfg, key, default=None):
    """Get config value by key with optional default."""
    return cfg.get(key, default)
def get_nested(cfg, path, default=None, sep="."):
    for k in path.split(sep):
        if not isinstance(cfg, dict): return default
        cfg = cfg.get(k, default)
    return cfg
def set_config_value(cfg, key, value): cfg[key] = value; return cfg
def delete_config_key(cfg, key): cfg.pop(key, None); return cfg
def config_keys(cfg):
    """Return list of config keys."""
    if not isinstance(cfg, dict):
        return []
    return list(cfg.keys())

def config_values(cfg):
    """Return list of config values."""
    if not isinstance(cfg, dict):
        return []
    return list(cfg.values())

def config_items(cfg):
    """Return list of (key, value) pairs."""
    if not isinstance(cfg, dict):
        return []
    return list(cfg.items())

def deep_copy_config(cfg):
    """Return deep copy of config."""
    import copy
    return copy.deepcopy(cfg)

def config_to_env(cfg, prefix=""):
    """Convert config to env-style dict (uppercase keys)."""
    if not isinstance(cfg, dict):
        return {}
    result = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}".upper()
        result[key] = str(v)
    return result

def env_to_config(env, prefix=""):
    """Convert env dict to config (keys starting with prefix, lowercased)."""
    result = {}
    for k, v in env.items():
        if k.startswith(prefix):
            key = k[len(prefix):].lower()
            result[key] = v
    return result

def load_env_config():
    """Load config from environment. Returns empty dict."""
    return {}

def config_override(base, overrides):
    """Merge overrides into base."""
    result = dict(base)
    for k, v in overrides.items():
        result[k] = v
    return result

def config_defaults(cfg, defaults):
    """Fill missing keys from defaults."""
    result = dict(defaults)
    for k, v in cfg.items():
        result[k] = v
    return result

def flatten_config(cfg, prefix=""):
    """Flatten nested config to dot-notation keys."""
    if not isinstance(cfg, dict):
        return {}
    result = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}" if prefix else k
        result[key] = v
    return result