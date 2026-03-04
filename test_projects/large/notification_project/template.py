"""Notification templates."""
def render_template(tpl, ctx):
    """Replace {{key}} with ctx[key]."""
    result = tpl
    for k, v in ctx.items():
        result = result.replace("{{" + k + "}}", str(v))
    return result

def validate_template(tpl, required):
    """Check all required placeholders exist."""
    for k in required:
        if "{{" + k + "}}" not in tpl:
            return False
    return True

def get_placeholders(tpl):
    """Extract placeholder names from template."""
    import re
    pat = r"\{\{\(\w+)\}\}"
    return re.findall(pat.replace("\\(", "(").replace("\\)", ")"), tpl)