"""Log formatting."""
def format_log(level, msg):
    """Format log line with level and message."""
    return f"[{level}] {msg}"

def format_with_time(level, msg, ts=""):
    """Format with optional timestamp prefix."""
    prefix = f"{ts} " if ts else ""
    return f"{prefix}[{level}] {msg}"

def format_json(level, msg, **kw):
    """Format as JSON object."""
    import json
    obj = {"level": level, "msg": msg}
    for k, v in kw.items():
        obj[k] = v
    return json.dumps(obj)
def format_syslog(level, msg, facility=16):
    sev = {"DEBUG":0,"INFO":1,"WARNING":2,"ERROR":3}.get(level, 1)
    return f"<{facility + sev}>{msg}"
def format_simple(level, msg):
    """Simple level: message format."""
    return f"{level}: {msg}"

def format_with_context(level, msg, ctx):
    """Format with context dict/str appended."""
    return f"[{level}] {msg} {ctx}"

def format_multiline(level, msg):
    """Prefix each line with level."""
    lines = msg.split("\n")
    result = []
    for line in lines:
        result.append(f"[{level}] {line}")
    return "\n".join(result)

def format_template(level, msg, template="{level} - {msg}"):
    """Format using template string."""
    return template.format(level=level, msg=msg)

def format_colored(level, msg):
    """Format (coloring not implemented, uses format_log)."""
    return format_log(level, msg)

def format_compact(level, msg):
    """Compact format: first letter of level + msg."""
    abbrev = level[:1] if level else ""
    return f"{abbrev}{msg}"

def format_iso(level, msg, ts=""):
    """Format with ISO timestamp."""
    prefix = f"{ts} " if ts else ""
    return f"{prefix}[{level}] {msg}"

def format_pid(level, msg, pid=0):
    """Format with process ID."""
    return f"[{pid}] [{level}] {msg}"

def format_module(level, msg, mod=""):
    """Format with module name."""
    return f"[{mod}] [{level}] {msg}"

def format_exception(level, msg, exc):
    """Format with exception."""
    return f"[{level}] {msg}: {exc}"

def format_traceback(level, msg, tb=""):
    """Format with traceback."""
    base = f"[{level}] {msg}"
    if tb:
        return base + "\n" + tb
    return base

def format_extra(level, msg, **extra):
    """Format with extra kwargs appended."""
    base = format_log(level, msg)
    if extra:
        return base + " " + str(extra)
    return base