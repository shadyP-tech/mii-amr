"""Navigation-layer facade for immutable JSON evidence artifacts.

The physical segment runner intentionally imports only navigation modules.
Keeping this tiny facade at that boundary preserves the runner's dependency
rule while delegating canonical serialization to the shared artifact store.
"""

from scripts.aufgabe04.artifacts.content_store import (
    payload_sha256,
    write_content_hashed_json,
)


__all__ = ["payload_sha256", "write_content_hashed_json"]
