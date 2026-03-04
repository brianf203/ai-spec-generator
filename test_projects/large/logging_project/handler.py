"""Log handling."""
def should_log(level, min_level):
    """Check if level meets minimum (by first char)."""
    if not level or not min_level:
        return True
    return ord(level[0]) >= ord(min_level[0])

def filter_logs(logs, level):
    """Filter logs containing level string."""
    result = []
    for l in logs:
        if level in l:
            result.append(l)
    return result

def filter_by_level(logs, min_level):
    """Filter logs by minimum level."""
    result = []
    for l in logs:
        extracted = ""
        if "[" in l:
            parts = l.split("[")
            if len(parts) > 1:
                extracted = parts[1].split("]")[0]
        if should_log(extracted, min_level):
            result.append(l)
    return result

def filter_by_pattern(logs, pattern):
    """Filter logs matching regex pattern."""
    import re
    result = []
    for l in logs:
        if re.search(pattern, l):
            result.append(l)
    return result

def filter_by_time(logs, start_ts, end_ts):
    """Filter by time range (stub, returns all)."""
    return logs

def aggregate_logs(logs):
    """Aggregate log stats."""
    return {"count": len(logs)}

def count_by_level(logs):
    """Count logs per level."""
    from collections import Counter
    levels = []
    for l in logs:
        if "[" in l:
            parts = l.split("[")[1].split("]")
            levels.append(parts[0] if parts else "")
        else:
            levels.append("")
    return dict(Counter(levels))

def log_contains(log, substr):
    """Check if log contains substring."""
    return substr in log

def log_level(log):
    """Extract level from log line."""
    if "[" not in log:
        return ""
    parts = log.split("[")[1].split("]")
    return parts[0] if parts else ""

def parse_log_line(line):
    """Parse log line to dict."""
    return {"raw": line}

def batch_logs(logs, size):
    """Split logs into batches."""
    result = []
    for i in range(0, len(logs), size):
        result.append(logs[i:i + size])
    return result

def dedupe_logs(logs):
    """Remove duplicate log lines."""
    seen = {}
    result = []
    for l in logs:
        if l not in seen:
            seen[l] = True
            result.append(l)
    return result

def truncate_logs(logs, max_len):
    """Truncate each log to max_len."""
    return [l[:max_len] for l in logs]

def sort_logs(logs, key=None):
    """Sort logs."""
    return sorted(logs, key=key)

def merge_log_streams(streams):
    """Merge multiple log streams."""
    result = []
    for s in streams:
        for l in s:
            result.append(l)
    return result

def log_sampling_rate(rate):
    """Return sampling rate (pass-through)."""
    return rate