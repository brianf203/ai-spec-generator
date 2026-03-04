"""Notification channels."""
def send_email(channel, to, subject, body):
    """Record email send. Returns dict."""
    channel.setdefault("emails", []).append({"to": to, "subject": subject, "body": body})
    return channel

def send_sms(channel, to, body):
    """Record SMS send."""
    channel.setdefault("sms", []).append({"to": to, "body": body})
    return channel

def send_push(channel, device_id, payload):
    """Record push notification."""
    channel.setdefault("push", []).append({"device": device_id, "payload": payload})
    return channel

def get_pending_emails(channel):
    """Return pending emails."""
    return channel.get("emails", [])

def get_pending_sms(channel):
    """Return pending SMS."""
    return channel.get("sms", [])

def clear_channel(channel):
    """Clear all pending."""
    channel.clear()
    return channel