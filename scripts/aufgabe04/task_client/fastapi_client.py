"""Small urllib-based FastAPI client for Aufgabe 04 task state."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import URLError
from urllib.parse import quote, urljoin
from urllib.request import urlopen

from .api_paths import ADMIN_ROBOT_PLANS_PATH, ADMIN_STATUS_PATH, HEALTH_PATH, OPENAPI_PATH
from .models import FastApiConfig


def _base_url(config: FastApiConfig) -> str:
    return config.base_url.rstrip("/") + "/"


def _get_json(config: FastApiConfig, path: str) -> Any:
    url = urljoin(_base_url(config), path.lstrip("/"))
    try:
        with urlopen(url, timeout=config.timeout_sec) as response:
            data = response.read().decode("utf-8")
    except URLError as exc:
        raise RuntimeError(f"FastAPI request failed for {url}: {exc}") from exc
    try:
        return json.loads(data)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"FastAPI response is not JSON for {url}") from exc


def health(config: FastApiConfig) -> Any:
    return _get_json(config, HEALTH_PATH)


def fetch_admin_status(config: FastApiConfig) -> Any:
    return _get_json(config, ADMIN_STATUS_PATH)


def fetch_robot_plans(config: FastApiConfig) -> Any:
    return _get_json(config, ADMIN_ROBOT_PLANS_PATH)


def fetch_openapi(config: FastApiConfig) -> Any:
    return _get_json(config, OPENAPI_PATH)


def report_scanned_qr(
    config: FastApiConfig,
    *,
    qr_id: str,
    station_id: str = "",
) -> Any:
    if not config.scanned_qr_endpoint_template:
        raise ValueError("scanned QR endpoint template is not configured")
    path = config.scanned_qr_endpoint_template.format(
        robot_id=quote(config.robot_id, safe=""),
        qr_id=quote(qr_id, safe=""),
        station_id=quote(station_id, safe=""),
    )
    return _get_json(config, path)
