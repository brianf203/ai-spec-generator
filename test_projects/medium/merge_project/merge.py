"""Merge sorted lists."""
def merge_sorted_lists(a, b):
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    for k in range(i, len(a)):
        out.append(a[k])
    for k in range(j, len(b)):
        out.append(b[k])
    return out
def merge_sorted_inplace(arr, lo, mid, hi):
    left = arr[lo:mid + 1]
    right = arr[mid + 1:hi + 1]
    i, j, k = 0, 0, lo
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1
    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1
    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1
def merge_k_sorted(lists):
    import heapq
    return list(heapq.merge(*lists))
def merge_with_duplicates(a, b):
    """Merge sorted lists and remove duplicates."""
    merged = merge_sorted_lists(a, b)
    seen = {}
    result = []
    for x in merged:
        if x not in seen:
            seen[x] = True
            result.append(x)
    return result
def merge_intervals(intervals):
    if not intervals:
        return []
    out = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
def merge_count_inversions(arr):
    cnt, _ = _merge_count(arr)
    return cnt
def _merge_count(arr):
    if len(arr) <= 1:
        return 0, arr
    mid = len(arr) // 2
    c1, left = _merge_count(arr[:mid])
    c2, right = _merge_count(arr[mid:])
    c3, merged = 0, []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
            c3 += len(left) - i
    for k in range(i, len(left)):
        merged.append(left[k])
    for k in range(j, len(right)):
        merged.append(right[k])
    return c1 + c2 + c3, merged