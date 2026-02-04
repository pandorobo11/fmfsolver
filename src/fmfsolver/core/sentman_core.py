from __future__ import annotations

import math

import numpy as np


def vhat_from_alpha_beta_stl(alpha_deg: float, beta_deg: float) -> np.ndarray:
    """Freestream unit vector in STL axes.

    Definitions:
      tan(alpha) = Vz / Vx
      tan(beta)  = Vy / Vx

    Convention fix:
      For V>0, beta>0 should correspond to wind blowing toward -Y (Vy negative).
      Therefore we set Vy ∝ -tan(beta).
    """
    a = math.radians(float(alpha_deg))
    b = math.radians(float(beta_deg))
    v = np.array([1.0, -math.tan(b), math.tan(a)], dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Invalid alpha/beta leading to zero direction.")
    return v / n


def sentman_dC_dA_vector(
    Vhat: np.ndarray,
    n_out: np.ndarray,
    S: float,
    Ti: float,
    Tw: float,
    Aref: float,
    shielded: bool = False,
) -> np.ndarray:
    """Compute Sentman dC/dA vector and angular factors.

    If shielded=True, include only the reflected term (C); otherwise include A+B+C.
    Returns (dC_dA, eta, gamma).
    """
    Vhat = np.asarray(Vhat, dtype=float)
    n_out = np.asarray(n_out, dtype=float)
    n_in = -n_out

    eta = float(np.dot(Vhat, n_in))
    gamma = math.sqrt(max(0.0, 1.0 - eta * eta))

    S = float(S)
    if S <= 0:
        raise ValueError(f"S must be > 0, got {S}")

    hs = eta * S
    Phi = 1.0 + math.erf(hs)
    E = math.exp(-(hs * hs))

    A = gamma * Phi + (1.0 / (S * math.sqrt(math.pi))) * E
    B = (1.0 / (2.0 * S * S)) * Phi
    C = (
        0.5
        * math.sqrt(float(Tw) / float(Ti))
        * ((eta * math.sqrt(math.pi) / S) * Phi + (1.0 / (S * S)) * E)
    )

    if shielded:
        dC_dA = (C * n_in) / float(Aref)
    else:
        dC_dA = (A * Vhat + (B + C) * n_in) / float(Aref)
    return dC_dA


def stl_to_body(v_stl: np.ndarray) -> np.ndarray:
    """Convert a vector from STL axes to body axes."""
    v = np.asarray(v_stl, dtype=float)
    return np.array([-v[0], v[1], -v[2]], dtype=float)


def rot_y(alpha_rad: float) -> np.ndarray:
    """Rotation matrix about +Y axis by alpha_rad (right-handed)."""
    c = math.cos(alpha_rad)
    s = math.sin(alpha_rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )
