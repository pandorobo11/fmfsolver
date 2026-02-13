from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import trimesh

import fmfsolver.core.shielding as shielding
from fmfsolver.core.shielding import clear_shield_cache, compute_shield_mask


class TestShieldingCache(unittest.TestCase):
    """Tests for in-process shield-mask cache behavior."""

    def setUp(self) -> None:
        clear_shield_cache()

    def test_reuses_mask_for_same_mesh_direction_and_batch(self) -> None:
        """Second call with same inputs should hit cache and skip ray query."""
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        centers = np.asarray(mesh.triangles_center, dtype=float)
        vhat = np.array([1.0, 0.0, 0.0], dtype=float)

        with (
            patch.object(shielding, "_SHIELD_CACHE_MAX", 1),
            patch.object(mesh.ray, "intersects_id", wraps=mesh.ray.intersects_id) as mock_ray,
        ):
            m1 = compute_shield_mask(mesh, centers, vhat, batch_size=8)
            m2 = compute_shield_mask(mesh, centers, vhat, batch_size=8)

        expected_batches = int(np.ceil(len(centers) / 8))
        self.assertEqual(mock_ray.call_count, expected_batches)
        self.assertTrue(np.array_equal(m1, m2))

    def test_direction_change_bypasses_cache(self) -> None:
        """Different freestream direction should compute a different cache key."""
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        centers = np.asarray(mesh.triangles_center, dtype=float)
        vhat_x = np.array([1.0, 0.0, 0.0], dtype=float)
        vhat_y = np.array([0.0, 1.0, 0.0], dtype=float)

        with (
            patch.object(shielding, "_SHIELD_CACHE_MAX", 1),
            patch.object(mesh.ray, "intersects_id", wraps=mesh.ray.intersects_id) as mock_ray,
        ):
            compute_shield_mask(mesh, centers, vhat_x, batch_size=8)
            compute_shield_mask(mesh, centers, vhat_y, batch_size=8)

        expected_batches = int(np.ceil(len(centers) / 8))
        self.assertEqual(mock_ray.call_count, expected_batches * 2)

    def test_three_sample_majority_vote(self) -> None:
        """Three-ray mode should use majority vote (2 of 3) per face."""

        class _DummyMesh:
            def __init__(self) -> None:
                self.faces = np.zeros((1, 3), dtype=int)
                self.bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
                self.triangles = np.array(
                    [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
                    dtype=float,
                )

        class _DummyIntersector:
            def intersects_id(self, **_kwargs):
                # rays 0 and 2 hit another face index => 2/3 majority true
                return np.array([1, 1], dtype=int), np.array([0, 2], dtype=int)

        mesh = _DummyMesh()
        centers = np.array([[1.0 / 3.0, 1.0 / 3.0, 0.0]], dtype=float)
        vhat = np.array([1.0, 0.0, 0.0], dtype=float)

        with patch.object(shielding, "_resolve_intersector", return_value=(_DummyIntersector(), "rtree")):
            mask = compute_shield_mask(
                mesh=mesh,
                centers_m=centers,
                Vhat=vhat,
                batch_size=8,
                ray_backend="rtree",
                shield_rays_mode="precise",
            )

        self.assertTrue(np.array_equal(mask, np.array([True], dtype=bool)))

    def test_three_sample_single_hit_is_not_shielded(self) -> None:
        """Three-ray mode requires at least two hits on other faces."""

        class _DummyMesh:
            def __init__(self) -> None:
                self.faces = np.zeros((1, 3), dtype=int)
                self.bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
                self.triangles = np.array(
                    [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
                    dtype=float,
                )

        class _DummyIntersector:
            def intersects_id(self, **_kwargs):
                # only one of three rays hits another face => vote fails
                return np.array([1], dtype=int), np.array([1], dtype=int)

        mesh = _DummyMesh()
        centers = np.array([[1.0 / 3.0, 1.0 / 3.0, 0.0]], dtype=float)
        vhat = np.array([1.0, 0.0, 0.0], dtype=float)

        with patch.object(shielding, "_resolve_intersector", return_value=(_DummyIntersector(), "rtree")):
            mask = compute_shield_mask(
                mesh=mesh,
                centers_m=centers,
                Vhat=vhat,
                batch_size=8,
                ray_backend="rtree",
                shield_rays_mode="precise",
            )

        self.assertTrue(np.array_equal(mask, np.array([False], dtype=bool)))


if __name__ == "__main__":
    unittest.main()
