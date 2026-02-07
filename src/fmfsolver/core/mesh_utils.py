from __future__ import annotations

"""Mesh loading and preprocessing helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import trimesh


@dataclass
class MeshData:
    """Preprocessed mesh and derived per-face geometric quantities.

    Attributes:
        mesh: Combined trimesh object after scaling and normal cleanup.
        centers_m: Triangle centers [m], shape ``(n_faces, 3)``.
        normals_out: Outward face normals (unit vectors), shape ``(n_faces, 3)``.
        areas_m2: Face areas [m^2], shape ``(n_faces,)``.
    """

    mesh: trimesh.Trimesh
    centers_m: np.ndarray
    normals_out: np.ndarray
    areas_m2: np.ndarray


def load_meshes(stl_paths: List[str], scale_m_per_unit: float, logfn) -> MeshData:
    """Load one or more STL meshes and compute face geometry.

    Args:
        stl_paths: STL file paths. Multiple files are concatenated into one mesh.
        scale_m_per_unit: Scalar conversion from STL units to meters.
        logfn: Logging callback accepting one message string.

    Returns:
        ``MeshData`` with scaled vertices, face centers, normals, and areas.

    Notes:
        - Scenes are flattened by concatenating all contained geometries.
        - Normal orientation is repaired when possible.
        - For watertight meshes with negative volume, orientation is inverted.
    """
    meshes = []
    for p in stl_paths:
        m = trimesh.load_mesh(Path(p).expanduser(), force="mesh")
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(tuple(m.geometry.values()))
        meshes.append(m)

    mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    mesh.vertices = mesh.vertices.astype(float) * float(scale_m_per_unit)

    # Always fix/check normals
    try:
        trimesh.repair.fix_normals(mesh)
    except Exception as e:
        logfn(f"[WARN] fix_normals failed: {e}")

    if mesh.is_watertight:
        try:
            if mesh.volume < 0:
                logfn("[WARN] Mesh volume negative -> inverting orientation (normals).")
                mesh.invert()
        except Exception as e:
            logfn(f"[WARN] volume/orientation check failed: {e}")
    else:
        logfn("[WARN] Mesh is not watertight (trimesh). Continuing anyway.")

    centers = mesh.triangles_center.astype(float)
    normals = mesh.face_normals.astype(float)
    areas = mesh.area_faces.astype(float)

    return MeshData(mesh=mesh, centers_m=centers, normals_out=normals, areas_m2=areas)
