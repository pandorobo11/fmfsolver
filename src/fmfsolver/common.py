"""Shared small helpers used across modules."""

from __future__ import annotations

import math


def is_filled(value) -> bool:
    """Return True when a table-like cell should be treated as specified."""
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip() != ""
