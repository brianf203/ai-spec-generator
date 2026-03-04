"""HTTP request parsing utilities."""
def parse_query_string(qs):
    """Parse query string into dict. Handles key=value pairs."""
    result = {}
    for part in qs.split("&"):
        if "=" in part:
            key, val = part.split("=", 1)
            result[key.strip()] = val.strip()
    return result

def get_header(headers, name):
    """Get header value by name. Case-sensitive."""
    return headers.get(name, "")

def get_header_lower(headers, name):
    """Get header, try lowercase name if not found."""
    if name in headers:
        return headers[name]
    return headers.get(name.lower(), "")

def parse_json_body(body):
    """Parse JSON body. Returns empty dict if body is empty."""
    if not body:
        return {}
    import json
    return json.loads(body)

def get_query_param(params, key, default=None):
    """Get query param with default."""
    return params.get(key, default)

def parse_content_type(ct):
    """Extract main content type from header (strip charset etc)."""
    if not ct:
        return ""
    return ct.split(";")[0].strip()

def is_json_request(headers):
    """Check if Content-Type indicates JSON."""
    ct = headers.get("Content-Type", "")
    return "application/json" in parse_content_type(ct)

def get_bearer_token(headers):
    """Extract Bearer token from Authorization header."""
    auth = headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return ""

def parse_cookies(cookie_str):
    """Parse Cookie header into dict."""
    if not cookie_str:
        return {}
    result = {}
    for part in cookie_str.split(";"):
        part = part.strip()
        if "=" in part:
            key, val = part.split("=", 1)
            result[key.strip()] = val.strip()
    return result

def get_client_ip(headers, default="0.0.0.0"):
    """Get client IP from X-Forwarded-For or X-Real-IP."""
    ip = headers.get("X-Forwarded-For", headers.get("X-Real-IP", default))
    return ip.split(",")[0].strip()

def validate_method(m):
    """Check HTTP method is allowed."""
    return m.upper() in ("GET", "POST", "PUT", "DELETE", "PATCH")

def parse_accept_header(accept):
    """Parse Accept header into list of types."""
    if not accept:
        return []
    return [x.strip().split(";")[0].strip() for x in accept.split(",")]

def get_user_agent(headers):
    """Get User-Agent header."""
    return headers.get("User-Agent", "")