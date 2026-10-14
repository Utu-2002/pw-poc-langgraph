import json
from pathlib import Path

USER_HISTORY_PATH = Path(__file__).parent.parent / "data" / "mock_user_history.json"
_USER_HISTORY = json.loads(USER_HISTORY_PATH.read_text(encoding="utf-8"))

_MOCK_POLICY = {
    "hardware": {"requires_extra_security_check": False},
    "software": {"requires_extra_security_check": False},
    "access": {"requires_extra_security_check": True},
    "network": {"requires_extra_security_check": False},
    "other": {"requires_extra_security_check": False},
}


def lookup_user_history(user_id: str) -> dict:
    past_requests = _USER_HISTORY.get(user_id, [])
    return {
        "past_requests_90d": len(past_requests),
        "past_requests_summary": past_requests,
    }


def lookup_policy(category: str) -> dict:
    return _MOCK_POLICY.get(category, {"requires_extra_security_check": False})


def create_ticket(priority: str, assigned_to: str = "IT_general") -> str:
    return f"[MOCK] Ticket created: priority={priority}, assigned_to={assigned_to}"


def send_notification(message: str) -> str:
    return f"[MOCK] Notification sent: {message}"