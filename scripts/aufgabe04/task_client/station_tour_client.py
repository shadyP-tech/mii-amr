"""Robot-scoped HTTP operations for a stored-pose station tour.

These paths and payloads follow the task server's RobotPlanRandomizeRequest and
QrScanRequest contracts. Mutating requests are attempted exactly once: a lost
response must be investigated using the persisted client event ID, not retried
with a fresh ID or followed by an automatic mission reset.
"""

from __future__ import annotations

import json
import math
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen


DEFAULT_SERVER_BASE_URL = "http://10.42.0.1:8000"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class StationTourHttpError(RuntimeError):
    def __init__(self, message, *, write_outcome_unknown=False, status_code=None, response=None):
        super().__init__(message)
        self.write_outcome_unknown = write_outcome_unknown
        self.status_code = status_code
        self.response = response


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a nonempty exact identifier")
    return value


class StationTourClient:
    def __init__(self, *, robot_id: str, base_url: str = DEFAULT_SERVER_BASE_URL,
                 timeout_sec: float = 3.0):
        self.robot_id = _text(robot_id, "robot_id")
        parsed = urlsplit(base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment):
            raise ValueError("server base URL must be an HTTP(S) URL without credentials, query or fragment")
        if type(timeout_sec) not in (int, float) or not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ValueError("HTTP timeout must be finite and positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = float(timeout_sec)

    @property
    def robot_path(self):
        return "/api/v1/robots/" + quote(self.robot_id, safe="")

    def _request(self, method, path, body=None):
        request = Request(
            self.base_url + path,
            data=None if body is None else json.dumps(body, allow_nan=False).encode("utf-8"),
            headers={"Accept": "application/json", **({} if body is None else {"Content-Type": "application/json"})},
            method=method,
        )
        mutating = method != "GET"
        try:
            with urlopen(request, timeout=self.timeout_sec) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            try:
                detail = exc.read(MAX_RESPONSE_BYTES).decode("utf-8", errors="replace")
            except (OSError, HTTPException):
                detail = "unreadable HTTP error response"
            finally:
                exc.close()
            raise StationTourHttpError(
                f"{method} {path} failed with HTTP {exc.code}",
                write_outcome_unknown=mutating and exc.code >= 500,
                status_code=exc.code, response=detail,
            ) from exc
        except (URLError, OSError, TimeoutError, HTTPException) as exc:
            raise StationTourHttpError(
                f"{method} {path} failed: {exc}", write_outcome_unknown=mutating,
            ) from exc
        try:
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError("response exceeds size limit")
            return json.loads(raw.decode("utf-8"), parse_constant=_invalid_constant, object_pairs_hook=_unique_object)
        except (UnicodeError, ValueError) as exc:
            raise StationTourHttpError(
                f"{method} {path} returned invalid JSON: {exc}",
                write_outcome_unknown=mutating,
            ) from exc

    def get_plan(self):
        return self._request("GET", self.robot_path + "/plan")

    def get_qr_mappings(self):
        return self._request("GET", self.robot_path + "/qr-mappings")

    def randomize_plan(self, *, qr_count: int, stations: int):
        if type(qr_count) is not int or not 4 <= qr_count <= 10:
            raise ValueError("qr_count must be an integer from 4 to 10")
        if type(stations) is not int or not 3 <= stations <= 100:
            raise ValueError("stations must be an integer from 3 to 100")
        return self._request("POST", self.robot_path + "/plan/randomize",
                             {"qr_count": qr_count, "stations": stations})

    def report_arrival(self, qr_id: str, *, client_event_id: str):
        return self._request("POST", "/api/v1/qr/" + quote(_text(qr_id, "qr_id"), safe="") + "/scan",
                             {"robot_id": self.robot_id,
                              "client_event_id": _text(client_event_id, "client_event_id")})


def _invalid_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
