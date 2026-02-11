from __future__ import annotations

"""Shielding (self-occlusion) evaluation for triangle panels."""

import os
import numpy as np
import trimesh


def _resolve_batch_size(batch_size: int | None) -> int:
    """Resolve shielding ray batch size from argument or environment."""
    if batch_size is None:
        raw = os.getenv("FMFSOLVER_SHIELD_BATCH_SIZE", "").strip()
        if raw:
            try:
                batch_size = int(raw)
            except ValueError as exc:
                raise ValueError(
                    "FMFSOLVER_SHIELD_BATCH_SIZE must be an integer >= 1."
                ) from exc
        else:
            batch_size = 8
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    return int(batch_size)


def compute_shield_mask(
    mesh: trimesh.Trimesh,
    centers_m: np.ndarray,
    Vhat: np.ndarray,
    batch_size: int | None = None,
) -> np.ndarray:
    """Return per-face shielding mask using one ray per face center.

    Rays are cast from each face center along ``-Vhat`` (upstream direction).
    A face is marked shielded if its ray first intersects another triangle
    (intersection triangle index differs from the source face index).

    Args:
        mesh: Combined triangle mesh in STL coordinates.
        centers_m: Face centers [m], shape ``(n_faces, 3)``.
        Vhat: Freestream direction vector in STL coordinates, shape ``(3,)``.
        batch_size: Number of rays processed per query batch. If omitted,
            uses ``FMFSOLVER_SHIELD_BATCH_SIZE`` when set, else ``8``.

    Returns:
        Boolean array of shape ``(n_faces,)`` where ``True`` means shielded.

    Notes:
        ``rtree`` is required by trimesh ray acceleration in this project.
    """
    batch_size = _resolve_batch_size(batch_size)

    Vhat = np.asarray(Vhat, dtype=float)
    Vn = float(np.linalg.norm(Vhat))
    if Vn == 0.0:
        raise ValueError("Vhat has zero norm.")
    d = -Vhat / Vn

    n_faces = int(len(centers_m))
    if n_faces == 0:
        return np.zeros(0, dtype=bool)

    bbox = mesh.bounds
    L = float(np.linalg.norm(bbox[1] - bbox[0]))
    eps = max(1e-9, 1e-6 * L)

    shielded = np.zeros(n_faces, dtype=bool)
    for start in range(0, n_faces, batch_size):
        end = min(start + batch_size, n_faces)
        origins = centers_m[start:end] + d[None, :] * eps
        directions = np.repeat(d[None, :], end - start, axis=0)

        tri_idx, ray_idx = mesh.ray.intersects_id(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=False,
            return_locations=False,
        )

        if len(ray_idx) == 0:
            continue

        ray_idx_global = ray_idx + start
        hit_other_face = tri_idx != ray_idx_global
        if np.any(hit_other_face):
            shielded[ray_idx_global[hit_other_face]] = True
    return shielded
