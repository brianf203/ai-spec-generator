"""Matrix utilities."""
def create_matrix(rows, cols, fill=0):
    """Create rows x cols matrix filled with fill."""
    return [[fill] * cols for _ in range(rows)]

def matrix_copy(m):
    """Deep copy of matrix."""
    return [row[:] for row in m]

def matrix_trace(m):
    """Sum of diagonal elements."""
    return sum(m[i][i] for i in range(min(len(m), len(m[0]))))

def matrix_diag(m):
    """Extract diagonal as list."""
    return [m[i][i] for i in range(min(len(m), len(m[0])))]