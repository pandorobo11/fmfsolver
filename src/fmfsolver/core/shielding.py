from __future__ import annotations

"""Shielding (self-occlusion) evaluation for triangle panels."""

import os
from collections import OrderedDict
from threading import Lock

import numpy as np
import trimesh
from trimesh.ray import has_embree, ray_triangle

try:
    from trimesh.ray import ray_pyembree
except Exception:  # pragma: no cover - depends on optional runtime dependency
    ray_pyembree = None

_SHIELD_CACHE_LOCK = Lock()
_SHIELD_MASK_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_INTERSECTOR_CACHE: OrderedDict[tuple, object] = OrderedDict()


def _resolve_shield_cache_max() -> int:
    """Return max number of cached shield masks kept in this process."""
    raw = os.getenv("FMFSOLVER_SHIELD_CACHE_MAX", "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("FMFSOLVER_SHIELD_CACHE_MAX must be an integer >= 0.") from exc
    if value < 0:
        raise ValueError("FMFSOLVER_SHIELD_CACHE_MAX must be >= 0.")
    return value


_SHIELD_CACHE_MAX = _resolve_shield_cache_max()


def _default_batch_size(ray_backend_resolved: str) -> int:
    """Return backend-aware default batch size."""
    if ray_backend_resolved == "embree":
        return 64
    return 8


def _normalize_ray_backend(ray_backend: str | None) -> str:
    """Normalize ray backend selector to one of auto/rtree/embree."""
    value = str(ray_backend or "auto").strip().lower()
    if value == "":
        value = "auto"
    if value not in {"auto", "rtree", "embree"}:
        raise ValueError("ray_backend must be one of: auto, rtree, embree.")
    return value


def _resolve_intersector(mesh: trimesh.Trimesh, ray_backend: str) -> tuple[object, str]:
    """Resolve ray intersector and report the effective backend name."""
    backend = _normalize_ray_backend(ray_backend)
    if backend == "auto":
        module = type(mesh.ray).__module__
        resolved = "embree" if "ray_pyembree" in module else "rtree"
        return mesh.ray, resolved

    key = (id(mesh), backend)
    with _SHIELD_CACHE_LOCK:
        cached = _INTERSECTOR_CACHE.get(key)
        if cached is not None:
            _INTERSECTOR_CACHE.move_to_end(key, last=True)
            return cached, backend

    if backend == "rtree":
        intersector = ray_triangle.RayMeshIntersector(mesh)
    else:
        if not has_embree or ray_pyembree is None:
            raise ValueError(
                "ray_backend='embree' was requested, but Embree is not available. "
                "Install optional dependency 'rayaccel' or use ray_backend='rtree'."
            )
        intersector = ray_pyembree.RayMeshIntersector(mesh)

    with _SHIELD_CACHE_LOCK:
        _INTERSECTOR_CACHE[key] = intersector
        _INTERSECTOR_CACHE.move_to_end(key, last=True)
        while len(_INTERSECTOR_CACHE) > 4:
            _INTERSECTOR_CACHE.popitem(last=False)
    return intersector, backend


def _resolve_batch_size(ray_backend_resolved: str, batch_size: int | None) -> int:
    """Resolve shielding ray batch size from argument, env, or backend."""
    if batch_size is None:
        raw = os.getenv("FMFSOLVER_SHIELD_BATCH_SIZE", "").strip()
        if raw:
            try:
                batch_size = int(raw)
            except ValueError as exc:
                raise ValueError("FMFSOLVER_SHIELD_BATCH_SIZE must be an integer >= 1.") from exc
        else:
            batch_size = _default_batch_size(ray_backend_resolved)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    return int(batch_size)


def _shield_cache_key(
    mesh: trimesh.Trimesh, Vhat: np.ndarray, batch_size: int, ray_backend_resolved: str
) -> tuple:
    """Build an in-process cache key for shield mask reuse."""
    Vhat = np.asarray(Vhat, dtype=float)
    norm = float(np.linalg.norm(Vhat))
    if norm == 0.0:
        raise ValueError("Vhat has zero norm.")
    d = -Vhat / norm
    # Rounded direction avoids tiny floating-point noise misses.
    d_key = tuple(np.round(d, decimals=12).tolist())
    return (id(mesh), int(len(mesh.faces)), int(batch_size), ray_backend_resolved, d_key)


def clear_shield_cache() -> None:
    """Clear in-process shield-mask cache."""
    with _SHIELD_CACHE_LOCK:
        _SHIELD_MASK_CACHE.clear()
        _INTERSECTOR_CACHE.clear()


def compute_shield_mask(
    mesh: trimesh.Trimesh,
    centers_m: np.ndarray,
    Vhat: np.ndarray,
    batch_size: int | None = None,
    ray_backend: str = "auto",
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
            uses ``FMFSOLVER_SHIELD_BATCH_SIZE`` when set, else backend-aware
            defaults (Embree: ``64``, rtree: ``8``).
        ray_backend: Ray intersector backend selector.
            - ``auto``: use trimesh default backend.
            - ``rtree``: force triangle intersector.
            - ``embree``: force Embree intersector.

    Returns:
        Boolean array of shape ``(n_faces,)`` where ``True`` means shielded.

    Notes:
        ``rtree`` is required by trimesh ray acceleration in this project.
    """
    shielded, _backend_used = compute_shield_mask_with_backend(
        mesh=mesh,
        centers_m=centers_m,
        Vhat=Vhat,
        batch_size=batch_size,
        ray_backend=ray_backend,
    )
    return shielded


def compute_shield_mask_with_backend(
    mesh: trimesh.Trimesh,
    centers_m: np.ndarray,
    Vhat: np.ndarray,
    batch_size: int | None = None,
    ray_backend: str = "auto",
) -> tuple[np.ndarray, str]:
    """Return shielding mask and the effective ray backend used.

    Args:
        mesh: Combined triangle mesh in STL coordinates.
        centers_m: Face centers [m], shape ``(n_faces, 3)``.
        Vhat: Freestream direction vector in STL coordinates, shape ``(3,)``.
        batch_size: Number of rays processed per query batch.
        ray_backend: Backend selector (``auto``/``rtree``/``embree``).

    Returns:
        Tuple ``(shielded, backend_used)`` where:
        - ``shielded`` is a bool array, shape ``(n_faces,)``.
        - ``backend_used`` is one of ``rtree`` or ``embree``.
    """
    intersector, ray_backend_resolved = _resolve_intersector(mesh, ray_backend)
    batch_size = _resolve_batch_size(ray_backend_resolved, batch_size)
    key = _shield_cache_key(mesh, Vhat, batch_size, ray_backend_resolved)
    if _SHIELD_CACHE_MAX > 0:
        with _SHIELD_CACHE_LOCK:
            cached = _SHIELD_MASK_CACHE.get(key)
            if cached is not None:
                _SHIELD_MASK_CACHE.move_to_end(key, last=True)
                return cached, ray_backend_resolved

    Vhat = np.asarray(Vhat, dtype=float)
    Vn = float(np.linalg.norm(Vhat))
    if Vn == 0.0:
        raise ValueError("Vhat has zero norm.")
    d = -Vhat / Vn

    n_faces = int(len(centers_m))
    if n_faces == 0:
        return np.zeros(0, dtype=bool), ray_backend_resolved

    bbox = mesh.bounds
    L = float(np.linalg.norm(bbox[1] - bbox[0]))
    eps = max(1e-9, 1e-6 * L)

    shielded = np.zeros(n_faces, dtype=bool)
    for start in range(0, n_faces, batch_size):
        end = min(start + batch_size, n_faces)
        origins = centers_m[start:end] + d[None, :] * eps
        directions = np.repeat(d[None, :], end - start, axis=0)

        tri_idx, ray_idx = intersector.intersects_id(
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

    if _SHIELD_CACHE_MAX > 0:
        with _SHIELD_CACHE_LOCK:
            _SHIELD_MASK_CACHE[key] = shielded
            _SHIELD_MASK_CACHE.move_to_end(key, last=True)
            while len(_SHIELD_MASK_CACHE) > _SHIELD_CACHE_MAX:
                _SHIELD_MASK_CACHE.popitem(last=False)
    return shielded, ray_backend_resolved
