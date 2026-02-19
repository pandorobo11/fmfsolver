"""Small UI-specific formatting helpers."""

from __future__ import annotations


def format_case_text(row: dict) -> str:
    """Build a compact, visualization-focused case summary for display."""
    parts: list[str] = []

    def add(k, v):
        if v is None:
            return
        s = str(v).strip()
        if s == "" or s.lower() == "nan":
            return
        parts.append(f"{k}={s}")

    add("case_id", row.get("case_id"))

    # Mode A/B inputs (as entered)
    if str(row.get("S")).strip() not in ("", "nan", "None") and str(
        row.get("Ti_K")
    ).strip() not in ("", "nan", "None"):
        add("mode", "A")
        add("S", row.get("S"))
        add("Ti", row.get("Ti_K"))
    elif str(row.get("Mach")).strip() not in ("", "nan", "None") and str(
        row.get("Altitude_km")
    ).strip() not in ("", "nan", "None"):
        add("mode", "B")
        add("Mach", row.get("Mach"))
        add("Alt_km", row.get("Altitude_km"))

    add("Tw", row.get("Tw_K"))
    add("alpha", row.get("alpha_deg"))
    add("beta", row.get("beta_deg"))
    add("shield", row.get("shielding_on"))
    add("ray", row.get("ray_backend"))
    return " | ".join(parts)
