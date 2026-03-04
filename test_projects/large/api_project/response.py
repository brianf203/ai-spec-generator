"""Response building."""
def json_response(data):
    """Build 200 JSON response."""
    return {"status": 200, "body": data}

def error_response(code, msg):
    """Build error response with status code and message."""
    return {"status": code, "error": msg}

def success_response(data, msg="OK"):
    """Build success response with data and message."""
    return {"status": 200, "data": data, "message": msg}

def created_response(data, location=""):
    """Build 201 created response with optional Location header."""
    resp = {"status": 201, "body": data}
    if location:
        resp["Location"] = location
    return resp

def no_content_response():
    """Build 204 no content response."""
    return {"status": 204}

def redirect_response(url, code=302):
    """Build redirect response."""
    return {"status": code, "Location": url}

def paginated_response(items, page, per_page, total):
    """Build paginated list response."""
    return {"status": 200, "items": items, "page": page, "per_page": per_page, "total": total}

def add_response_header(resp, name, value):
    """Add header to response dict."""
    resp[name] = value
    return resp

def set_response_status(resp, code):
    """Set status code on response."""
    resp["status"] = code
    return resp

def wrap_error(e):
    """Wrap exception as 500 error response."""
    return error_response(500, str(e))

def cors_headers(origin="*"):
    """Build CORS headers."""
    return {"Access-Control-Allow-Origin": origin}

def cache_headers(max_age=3600):
    """Build Cache-Control header."""
    return {"Cache-Control": f"max-age={max_age}"}

def etag_header(etag):
    """Build ETag header."""
    return {"ETag": etag}