"""Path utilities."""
def join_path(*parts):
    """Join path parts."""
    return "/".join(p for p in parts if p)

def split_path(path):
    """Split path into parts."""
    return [p for p in path.split("/") if p]

def basename(path):
    """Get last path component."""
    parts = split_path(path)
    return parts[-1] if parts else ""

def dirname(path):
    """Get path without last component."""
    parts = split_path(path)
    return "/".join(parts[:-1]) if len(parts) > 1 else ""