"""File metadata."""
def create_metadata(path, size, mtime):
    """Create metadata dict."""
    return {"path": path, "size": size, "mtime": mtime}

def get_path(m):
    """Get path from metadata."""
    return m["path"]

def get_size(m):
    """Get size."""
    return m["size"]

def get_mtime(m):
    """Get mtime."""
    return m["mtime"]

def metadata_eq(a, b):
    """Compare metadata."""
    return a.get("path") == b.get("path") and a.get("size") == b.get("size")