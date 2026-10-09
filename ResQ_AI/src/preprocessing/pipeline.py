import re


def preprocess_text(text: str) -> str:
    """Normalize emergency text for downstream ML and rule-based logic."""
    if text is None:
        return "emergency incident"
    cleaned = str(text).lower().strip()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "emergency incident"
    return cleaned
