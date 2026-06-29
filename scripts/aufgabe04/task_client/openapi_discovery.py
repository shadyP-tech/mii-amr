"""Conservative OpenAPI scan-endpoint discovery."""

from __future__ import annotations

from typing import Mapping, Sequence


def discover_scan_endpoint_template(openapi_payload: Mapping[str, object]) -> str:
    paths = openapi_payload.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("OpenAPI payload does not contain paths")
    candidates = []
    for path, methods in paths.items():
        if not isinstance(path, str) or not isinstance(methods, Mapping):
            continue
        operation = methods.get("get")
        if not isinstance(operation, Mapping):
            continue
        haystack = " ".join(
            str(operation.get(key, ""))
            for key in ("operationId", "summary", "description")
        ).lower()
        parameters = _parameters(operation.get("parameters"))
        parameter_names = {parameter[0] for parameter in parameters}
        path_lower = path.lower()
        if ("qr" in path_lower or "scan" in path_lower or "qr" in haystack or "scan" in haystack) and (
            "qr" in parameter_names or "qr_id" in parameter_names or "station_id" in parameter_names
        ):
            candidates.append(_endpoint_template(path, parameters))
    if not candidates:
        raise ValueError("no GET scan endpoint found in OpenAPI")
    if len(candidates) > 1:
        raise ValueError(f"multiple possible GET scan endpoints found: {', '.join(sorted(candidates))}")
    return candidates[0]


def _parameters(parameters: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(parameters, Sequence) or isinstance(parameters, (str, bytes)):
        return ()
    names = []
    for item in parameters:
        if isinstance(item, Mapping):
            name = item.get("name")
            location = item.get("in")
            if isinstance(name, str):
                names.append((name.lower(), location if isinstance(location, str) else ""))
    return tuple(names)


def _endpoint_template(path: str, parameters: tuple[tuple[str, str], ...]) -> str:
    query_items = []
    for name, location in parameters:
        if location == "query" and name in {"qr", "qr_id", "station_id"}:
            value_name = "qr_id" if name in {"qr", "qr_id"} else "station_id"
            query_items.append(f"{name}={{{value_name}}}")
    if not query_items:
        return path
    separator = "&" if "?" in path else "?"
    return path + separator + "&".join(query_items)
