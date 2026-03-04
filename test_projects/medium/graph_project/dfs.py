"""DFS."""
def dfs(graph, start, visited=None):
    visited = visited or set()
    visited.add(start)
    for n in graph.get(start, []):
        if n not in visited: dfs(graph, n, visited)
    return list(visited)
def dfs_iterative(graph, start):
    stack, visited = [start], set()
    while stack:
        v = stack.pop()
        if v not in visited:
            visited.add(v)
            stack.extend(graph.get(v, []))
    return list(visited)
def dfs_path(graph, start, end, path=None):
    path = path or []
    path.append(start)
    if start == end: return path[:]
    for n in graph.get(start, []):
        if n not in path:
            p = dfs_path(graph, n, end, path)
            if p: return p
    path.pop()
    return None
def dfs_cycle(graph):
    GRAY, BLACK = 1, 2
    color = {}
    def visit(v):
        color[v] = GRAY
        for n in graph.get(v, []):
            if n not in color:
                if visit(n): return True
            elif color[n] == GRAY: return True
        color[v] = BLACK
        return False
    for v in graph:
        if v not in color and visit(v): return True
    return False
def topological_sort(graph):
    visited, result = set(), []
    def visit(v):
        visited.add(v)
        for n in graph.get(v, []):
            if n not in visited: visit(n)
        result.append(v)
    for v in graph:
        if v not in visited: visit(v)
    return result[::-1]
def count_components(graph):
    visited, count = set(), 0
    def dfs(v):
        visited.add(v)
        for n in graph.get(v, []):
            if n not in visited: dfs(n)
    for v in graph:
        if v not in visited: dfs(v); count += 1
    return count