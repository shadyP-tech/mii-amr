"""Case-preserving identifiers at the station/server boundary.

Station IDs and QR payloads are opaque identifiers. Only surrounding transport
whitespace is removed; case folding or constructing a station name from a QR
would silently change the server's identity contract.
"""

from __future__ import annotations

import re

_STATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_QR_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def canonical_server_station_id(value: str) -> str:
    if not isinstance(value, str) or not _STATION_ID.fullmatch(value.strip()):
        raise ValueError("server station ID must be a nonempty safe identifier")
    return value.strip()


def canonical_qr_id(value: str) -> str:
    if isinstance(value, str) and not value.strip():
        raise ValueError("QR payload is empty")
    if not isinstance(value, str) or not _QR_ID.fullmatch(value.strip()):
        raise ValueError("QR payload must be a single station or QR identifier")
    return value.strip()
