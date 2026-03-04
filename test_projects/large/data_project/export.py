"""Export."""
def to_csv_row(d, keys):
    """Convert dict to CSV row string."""
    parts = []
    for k in keys:
        parts.append(str(d.get(k, "")))
    return ",".join(parts)

def from_csv_row(row, keys):
    """Parse CSV row into dict."""
    values = row.split(",")
    result = {}
    for i, k in enumerate(keys):
        if i < len(values):
            result[k] = values[i]
    return result

def to_csv(rows, keys):
    """Convert list of dicts to CSV string."""
    lines = []
    for r in rows:
        lines.append(to_csv_row(r, keys))
    return "\n".join(lines)

def from_csv(csv_str, keys):
    """Parse CSV string into list of dicts."""
    result = []
    for line in csv_str.strip().split("\n"):
        if line:
            result.append(from_csv_row(line, keys))
    return result

def to_json(obj):
    """Serialize object to JSON string."""
    import json
    return json.dumps(obj)

def from_json(s):
    """Parse JSON string to object."""
    import json
    return json.loads(s)

def to_dict_list(rows, keys):
    """Extract specified keys from each row."""
    from transform import pick
    result = []
    for r in rows:
        result.append(pick(r, keys))
    return result

def escape_csv_val(v):
    """Escape value for CSV (quote if contains comma or quote)."""
    s = str(v)
    if "," in s or chr(34) in s:
        escaped = s.replace(chr(34), chr(34) + chr(34))
        return f'"{escaped}"'
    return s

def parse_csv_line(line):
    """Parse CSV line into list of values."""
    result = []
    for x in line.split(","):
        result.append(x.strip('"'))
    return result

def csv_headers(rows):
    """Get headers from first row. Empty list if no rows."""
    if not rows:
        return []
    return list(rows[0].keys())