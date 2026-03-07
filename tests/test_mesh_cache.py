from __future__ import annotations

import unittest

import numpy as np

from fmfsolver.core.mesh_utils import clear_mesh_cache, load_meshes, mesh_cache_stats


class TestMeshCache(unittest.TestCase):
    def test_load_meshes_reuses_cached_entry_for_same_input(self):
        clear_mesh_cache(reset_stats=True)
        stl = "samples/stl/cube.stl"

        md1 = load_meshes([stl], 1.0, lambda _msg: None)
        md2 = load_meshes([stl], 1.0, lambda _msg: None)
        stats = mesh_cache_stats()

        self.assertIsNot(md1, md2)
        self.assertEqual(stats.misses, 1)
        self.assertEqual(stats.hits, 1)
        self.assertEqual(stats.entries, 1)

    def test_load_meshes_cache_key_includes_scale(self):
        clear_mesh_cache(reset_stats=True)
        stl = "samples/stl/cube.stl"

        _ = load_meshes([stl], 1.0, lambda _msg: None)
        _ = load_meshes([stl], 0.001, lambda _msg: None)
        stats = mesh_cache_stats()

        self.assertEqual(stats.misses, 2)
        self.assertEqual(stats.hits, 0)
        self.assertEqual(stats.entries, 1)

    def test_load_meshes_cached_result_is_not_mutated_by_caller(self):
        clear_mesh_cache(reset_stats=True)
        stl = "samples/stl/cube.stl"

        md1 = load_meshes([stl], 1.0, lambda _msg: None)
        expected_centers = md1.centers_m.copy()
        expected_vertices = md1.mesh.vertices.copy()

        md1.centers_m[0, 0] = 123.0
        md1.mesh.vertices[0, 0] = 456.0

        md2 = load_meshes([stl], 1.0, lambda _msg: None)

        np.testing.assert_allclose(md2.centers_m, expected_centers)
        np.testing.assert_allclose(md2.mesh.vertices, expected_vertices)


if __name__ == "__main__":
    unittest.main()
