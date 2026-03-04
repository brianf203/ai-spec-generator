"""File index."""
def add_to_index(idx, path, meta):
    """Add file to index."""
    idx[path] = meta
    return idx

def remove_from_index(idx, path):
    """Remove from index."""
    idx.pop(path, None)
    return idx

def lookup_index(idx, path):
    """Lookup by path."""
    return idx.get(path)

def index_paths(idx):
    """Return all indexed paths."""
    return list(idx.keys())