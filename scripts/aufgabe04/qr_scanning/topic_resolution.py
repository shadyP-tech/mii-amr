"""Topic normalization for Aufgabe 04 QR camera subscribers."""

from __future__ import annotations


DEFAULT_COMPRESSED_IMAGE_TOPIC = "camera/image_raw/compressed"


def _clean_namespace(namespace: str | None) -> str:
    if not namespace:
        return ""
    return namespace.strip().strip("/")


def resolve_topic(topic: str, namespace: str | None = None) -> str:
    text = topic.strip()
    if not text:
        raise ValueError("topic must not be empty")
    if text.startswith("/"):
        return "/" + text.strip("/")

    relative = text.strip("/")
    ns = _clean_namespace(namespace)
    if ns:
        return f"/{ns}/{relative}"
    return f"/{relative}"
