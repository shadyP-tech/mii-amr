#!/usr/bin/env python3
"""
Compatibility facade for rectangular-arena geometry localization helpers.

The implementation lives in arena_geometry_localization. This module preserves
historical imports used by runtime scripts, tests, and offline diagnostics.
"""

from arena_geometry_localization import *


__all__ = [name for name in globals() if not name.startswith("_")]
