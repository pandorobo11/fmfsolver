from __future__ import annotations

"""Sentman free-molecular-flow core equations and coordinate transforms."""

import math

import numpy as np


def vhat_from_alpha_beta_stl(alpha_deg: float, beta_deg: float) -> np.ndarray:
    """Return freestream unit vector in STL axes from aerodynamic angles.

    Definitions:
      tan(alpha) = Vz / Vx
      tan(beta)  = Vy / Vx

    Convention fix:
      For V>0, beta>0 should correspond to wind blowing toward -Y (Vy negative).
      Therefore we set Vy ∝ -tan(beta).

    Args:
        alpha_deg: Angle of attack in degrees.
        beta_deg: Sideslip angle in degrees.

    Returns:
        Normalized freestream direction vector in STL axes, shape ``(3,)``.
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
    """Compute panel force-coefficient density vector ``dC/dA``.

    The implemented formula follows the codebase's Sentman convention:
    ``dC/dA = (A*Vhat + (B + C)*n_in) / Aref``.

    When ``shielded=True``, the panel contributes zero force and the function
    returns a zero vector immediately.

    Args:
        Vhat: Freestream unit vector in STL axes, shape ``(3,)``.
        n_out: Outward panel normal in STL axes, shape ``(3,)``.
        S: Molecular speed ratio.
        Ti: Free-stream translational temperature [K].
        Tw: Wall temperature [K].
        Aref: Reference area [m^2] for non-dimensionalization.
        shielded: If true, skip force evaluation and return zero.

    Returns:
        Force-coefficient density vector in STL axes, shape ``(3,)``.
    """
    if shielded:
        return np.zeros(3, dtype=float)

    Vhat = np.asarray(Vhat, dtype=float)
    n_out = np.asarray(n_out, dtype=float)
    n_in = -n_out

    gamma = float(np.dot(Vhat, n_in))

    S = float(S)
    if S <= 0:
        raise ValueError(f"S must be > 0, got {S}")

    hs = gamma * S
    Phi = 1.0 + math.erf(hs)
    E = math.exp(-(hs * hs))

    A = gamma * Phi + (1.0 / (S * math.sqrt(math.pi))) * E
    B = (1.0 / (2.0 * S * S)) * Phi
    C = (
        0.5
        * math.sqrt(float(Tw) / float(Ti))
        * ((gamma * math.sqrt(math.pi) / S) * Phi + (1.0 / (S * S)) * E)
    )

    dC_dA = (A * Vhat + (B + C) * n_in) / float(Aref)
    return dC_dA


def stl_to_body(v_stl: np.ndarray) -> np.ndarray:
    """Convert a vector from STL axes to body axes.

    Axis mapping is ``body = (-x_stl, +y_stl, -z_stl)``.
    """
    v = np.asarray(v_stl, dtype=float)
    return np.array([-v[0], v[1], -v[2]], dtype=float)


def rot_y(alpha_rad: float) -> np.ndarray:
    """Return right-handed rotation matrix about +Y by ``alpha_rad``.

    Args:
        alpha_rad: Rotation angle [rad].

    Returns:
        3x3 rotation matrix.
    """
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
