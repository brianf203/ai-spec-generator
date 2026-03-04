"""Matrix operations."""
def matrix_rows(m):
    """Return number of rows."""
    return len(m)

def matrix_cols(m):
    """Return number of columns. 0 if empty."""
    if not m:
        return 0
    return len(m[0])

def matrix_get(m, i, j):
    """Get element at (i, j)."""
    return m[i][j]

def matrix_set(m, i, j, val):
    """Set element at (i, j)."""
    m[i][j] = val
    return m

def matrix_add(a, b):
    """Add two matrices. Same dimensions required."""
    rows, cols = len(a), len(a[0])
    result = [[a[i][j] + b[i][j] for j in range(cols)] for i in range(rows)]
    return result

def matrix_transpose(m):
    """Transpose matrix."""
    if not m:
        return []
    rows, cols = len(m), len(m[0])
    return [[m[i][j] for i in range(rows)] for j in range(cols)]

def matrix_multiply(a, b):
    """Multiply matrices. a cols must equal b rows."""
    ra, ca, cb = len(a), len(a[0]), len(b[0])
    result = [[0] * cb for _ in range(ra)]
    for i in range(ra):
        for j in range(cb):
            total = 0
            for k in range(ca):
                total += a[i][k] * b[k][j]
            result[i][j] = total
    return result