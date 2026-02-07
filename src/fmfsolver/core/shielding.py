from __future__ import annotations

"""Shielding (self-occlusion) evaluation for triangle panels."""

import numpy as np
import trimesh


def compute_shield_mask(
    mesh: trimesh.Trimesh, centers_m: np.ndarray, Vhat: np.ndarray
) -> np.ndarray:
    """Return per-face shielding mask using one ray per face center.

    Rays are cast from each face center along ``-Vhat`` (upstream direction).
    A face is marked shielded if its ray first intersects another triangle
    (intersection triangle index differs from the source face index).

    Args:
        mesh: Combined triangle mesh in STL coordinates.
        centers_m: Face centers [m], shape ``(n_faces, 3)``.
        Vhat: Freestream direction vector in STL coordinates, shape ``(3,)``.

    Returns:
        Boolean array of shape ``(n_faces,)`` where ``True`` means shielded.

    Notes:
        ``rtree`` is required by trimesh ray acceleration in this project.
    """
    Vhat = np.asarray(Vhat, dtype=float)
    Vn = float(np.linalg.norm(Vhat))
    if Vn == 0.0:
        raise ValueError("Vhat has zero norm.")
    d = -Vhat / Vn

    bbox = mesh.bounds
    L = float(np.linalg.norm(bbox[1] - bbox[0]))
    eps = max(1e-9, 1e-6 * L)

    origins = centers_m + d[None, :] * eps
    directions = np.repeat(d[None, :], len(centers_m), axis=0)

    tri_idx, ray_idx, _loc = mesh.ray.intersects_id(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=False,
        return_locations=True,
    )

    shielded = np.zeros(len(centers_m), dtype=bool)
    for t, r in zip(tri_idx, ray_idx):
        if int(t) != int(r):
            shielded[int(r)] = True
    return shielded
