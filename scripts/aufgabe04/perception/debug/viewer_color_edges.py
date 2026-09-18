"""Compatibility exports for the shared stand colour support."""

from scripts.aufgabe04.perception.stand_color_support import (
    STAND_EDGE_PALETTE as VIEWER_STAND_PALETTE, color_edge_support, color_edge_exclusion,
)

__all__ = ["VIEWER_STAND_PALETTE", "color_edge_support", "color_edge_exclusion"]
