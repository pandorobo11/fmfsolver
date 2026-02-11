from __future__ import annotations

"""Mesh loading and preprocessing helpers."""

from dataclasses import dataclass
from threading import Lock
from pathlib import Path
from typing import List
from collections import OrderedDict

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
        face_stl_index: Source STL index per face, shape ``(n_faces,)``.
        stl_paths_order: Absolute STL paths in concatenation order.
    """

    mesh: trimesh.Trimesh
    centers_m: np.ndarray
    normals_out: np.ndarray
    areas_m2: np.ndarray
    face_stl_index: np.ndarray
    stl_paths_order: tuple[str, ...]


@dataclass(frozen=True)
class MeshCacheStats:
    """Runtime statistics for the in-process mesh cache."""

    entries: int
    hits: int
    misses: int


_CACHE_LOCK = Lock()
_MESH_CACHE: OrderedDict[tuple, MeshData] = OrderedDict()
_CACHE_HITS = 0
_CACHE_MISSES = 0
_MESH_CACHE_MAX = 1


def _mesh_cache_key(stl_paths: List[str], scale_m_per_unit: float) -> tuple:
    """Build a cache key from absolute path metadata and scale."""
    key_paths = []
    for p in stl_paths:
        path = Path(p).expanduser().resolve()
        st = path.stat()
        key_paths.append((str(path), int(st.st_size), int(st.st_mtime_ns)))
    return (tuple(key_paths), float(scale_m_per_unit))


def _load_meshes_uncached(stl_paths: List[str], scale_m_per_unit: float, logfn) -> MeshData:
    """Load mesh data from disk and compute per-face geometric arrays."""
    meshes = []
    stl_paths_order = tuple(str(Path(p).expanduser().resolve()) for p in stl_paths)
    for p in stl_paths:
        m = trimesh.load_mesh(Path(p).expanduser(), force="mesh")
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(tuple(m.geometry.values()))
        meshes.append(m)

    face_stl_index = np.concatenate(
        [
            np.full(len(m.faces), i, dtype=np.int32)
            for i, m in enumerate(meshes)
        ]
    )

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
    return MeshData(
        mesh=mesh,
        centers_m=centers,
        normals_out=normals,
        areas_m2=areas,
        face_stl_index=face_stl_index,
        stl_paths_order=stl_paths_order,
    )


def clear_mesh_cache(reset_stats: bool = True) -> None:
    """Clear in-process mesh cache; optionally reset hit/miss counters."""
    global _CACHE_HITS, _CACHE_MISSES
    with _CACHE_LOCK:
        _MESH_CACHE.clear()
        if reset_stats:
            _CACHE_HITS = 0
            _CACHE_MISSES = 0


def mesh_cache_stats() -> MeshCacheStats:
    """Return current cache entry count and hit/miss counters."""
    with _CACHE_LOCK:
        return MeshCacheStats(
            entries=len(_MESH_CACHE),
            hits=_CACHE_HITS,
            misses=_CACHE_MISSES,
        )


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
        - Results are cached by ``(paths, file metadata, scale)`` within the process.
    """
    global _CACHE_HITS, _CACHE_MISSES
    key = _mesh_cache_key(stl_paths, scale_m_per_unit)
    with _CACHE_LOCK:
        cached = _MESH_CACHE.get(key)
        if cached is not None:
            _CACHE_HITS += 1
            _MESH_CACHE.move_to_end(key, last=True)
            return cached

    loaded = _load_meshes_uncached(stl_paths, scale_m_per_unit, logfn)
    with _CACHE_LOCK:
        _CACHE_MISSES += 1
        _MESH_CACHE[key] = loaded
        _MESH_CACHE.move_to_end(key, last=True)
        while len(_MESH_CACHE) > _MESH_CACHE_MAX:
            _MESH_CACHE.popitem(last=False)
    return loaded
