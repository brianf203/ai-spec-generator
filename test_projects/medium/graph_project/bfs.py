"""BFS."""
def bfs(graph, start):
    from collections import deque
    visited, q = set(), deque([start])
    while q:
        v = q.popleft()
        if v not in visited:
            visited.add(v)
            q.extend(graph.get(v, []))
    return list(visited)
def bfs_levels(graph, start):
    from collections import deque
    visited, q, levels = {start}, deque([(start, 0)]), {start: 0}
    while q:
        v, d = q.popleft()
        for n in graph.get(v, []):
            if n not in visited:
                visited.add(n); levels[n] = d + 1; q.append((n, d + 1))
    return levels
def bfs_path(graph, start, end):
    from collections import deque
    if start == end: return [start]
    q, parent = deque([start]), {start: None}
    while q:
        v = q.popleft()
        for n in graph.get(v, []):
            if n not in parent: parent[n] = v; q.append(n)
            if n == end:
                path, cur = [], end
                while cur: path.append(cur); cur = parent[cur]
                return path[::-1]
    return []
def bfs_bipartite(graph):
    color, q = {}, []
    for s in graph:
        if s in color: continue
        color[s], q = 0, [s]
        while q:
            v, q = q[0], q[1:]
            for n in graph.get(v, []):
                if n not in color: color[n] = 1 - color[v]; q.append(n)
                elif color[n] == color[v]: return False
    return True