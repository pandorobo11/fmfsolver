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

    def test_cached_mask_is_not_mutated_by_caller(self) -> None:
        """Caller-side mutation should not affect future cache hits."""
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        centers = np.asarray(mesh.triangles_center, dtype=float)
        vhat = np.array([1.0, 0.0, 0.0], dtype=float)

        with patch.object(shielding, "_SHIELD_CACHE_MAX", 1):
            first = compute_shield_mask(mesh, centers, vhat, batch_size=8)
            expected = first.copy()
            first[:] = ~first
            second = compute_shield_mask(mesh, centers, vhat, batch_size=8)

        self.assertTrue(np.array_equal(second, expected))


if __name__ == "__main__":
    unittest.main()
