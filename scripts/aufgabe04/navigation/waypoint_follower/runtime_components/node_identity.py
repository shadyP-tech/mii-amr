"""Pure ROS graph-identity formatting used by runtime ownership checks."""

from __future__ import annotations


def node_identity(endpoint) -> str:
    namespace = getattr(endpoint, "node_namespace", "") or ""
    name = getattr(endpoint, "node_name", "") or ""
    return format_node_identity(namespace, name)


def format_node_identity(namespace: str, name: str) -> str:
    if namespace in ("", "/"):
        return f"/{name}"
    return f"{namespace.rstrip('/')}/{name}"
