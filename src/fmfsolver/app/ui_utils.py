"""Small UI-specific formatting helpers."""

from __future__ import annotations

def _as_float(value):
    """Parse value to float when possible."""
    try:
        if value is None:
            return None
        s = str(value).strip()
        if s == "" or s.lower() == "nan":
            return None
        return float(s)
    except Exception:
        return None


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

    alpha_in = _as_float(row.get("alpha_deg"))
    beta_in = _as_float(row.get("beta_or_bank_deg"))
    att = str(row.get("attitude_input") or "").strip().lower() or "beta_tan"
    if att == "beta_sin":
        add("alpha_t", alpha_in)
        add("beta_s", beta_in)
    elif att == "bank":
        add("alpha_i", alpha_in)
        add("phi", beta_in)
    else:
        add("alpha_t", alpha_in)
        add("beta_t", beta_in)

    add("shield", row.get("shielding_on"))
    add("ray", row.get("ray_backend"))
    return " | ".join(parts)
