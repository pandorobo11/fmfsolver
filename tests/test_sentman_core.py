from __future__ import annotations

import math
import unittest

import numpy as np

from fmfsolver.core.sentman_core import (
    resolve_attitude_to_vhat,
    sentman_dC_dA_vector,
    sentman_dC_dA_vectors,
    stl_to_body,
)


class TestSentmanCore(unittest.TestCase):
    def test_vhat_matches_equivalent_tan_form(self):
        angles = [(-70.0, -30.0), (-25.0, 10.0), (0.0, 0.0), (35.0, -15.0), (70.0, 30.0)]
        for alpha_deg, beta_deg in angles:
            v, _, _, mode = resolve_attitude_to_vhat(alpha_deg, beta_deg, "beta_tan")

            a = math.radians(alpha_deg)
            b = math.radians(beta_deg)
            v_ref = np.array([1.0, -math.tan(b), math.tan(a)], dtype=float)
            v_ref /= np.linalg.norm(v_ref)

            self.assertEqual(mode, "beta_tan")
            np.testing.assert_allclose(v, v_ref, rtol=0.0, atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=12)

    def test_sentman_vector_is_zero_when_shielded(self):
        v = sentman_dC_dA_vector(
            Vhat=np.array([1.0, 0.0, 0.0]),
            n_out=np.array([-1.0, 0.0, 0.0]),
            S=5.0,
            Ti=300.0,
            Tw=300.0,
            Aref=1.0,
            shielded=True,
        )
        np.testing.assert_allclose(v, np.zeros(3), rtol=0.0, atol=0.0)

    def test_stl_to_body_axis_mapping(self):
        v_stl = np.array([2.0, -3.0, 4.5], dtype=float)
        v_body = stl_to_body(v_stl)
        np.testing.assert_allclose(v_body, np.array([-2.0, -3.0, -4.5]), rtol=0.0, atol=0.0)

    def test_sentman_vectors_matches_scalar(self):
        Vhat, _, _, _ = resolve_attitude_to_vhat(10.0, -5.0, "beta_tan")
        n_out = np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        shielded = np.array([False, True, False, False], dtype=bool)
        vec = sentman_dC_dA_vectors(
            Vhat=Vhat,
            n_out=n_out,
            S=5.0,
            Ti=300.0,
            Tw=450.0,
            Aref=2.0,
            shielded=shielded,
        )
        ref = np.vstack(
            [
                sentman_dC_dA_vector(
                    Vhat=Vhat,
                    n_out=n_out[i],
                    S=5.0,
                    Ti=300.0,
                    Tw=450.0,
                    Aref=2.0,
                    shielded=bool(shielded[i]),
                )
                for i in range(n_out.shape[0])
            ]
        )
        np.testing.assert_allclose(vec, ref, rtol=0.0, atol=1e-13)

    def test_resolve_attitude_beta_tan_matches_direct(self):
        v1, a_t, b_t, mode = resolve_attitude_to_vhat(10.0, -5.0, "beta_tan")
        v2, _, _, _ = resolve_attitude_to_vhat(10.0, -5.0, "")
        self.assertEqual(mode, "beta_tan")
        self.assertAlmostEqual(a_t, 10.0, places=12)
        self.assertAlmostEqual(b_t, -5.0, places=12)
        np.testing.assert_allclose(v1, v2, rtol=0.0, atol=1e-12)

    def test_resolve_attitude_bank_and_beta_sin(self):
        # Ground truth from bank definition.
        v_ref, _, _, _ = resolve_attitude_to_vhat(30.0, 25.0, "bank")

        # Build equivalent mixed (alpha_t, beta_s) input from the same vector.
        alpha_t = math.degrees(math.atan2(float(v_ref[2]), float(v_ref[0])))
        beta_s = math.degrees(math.asin(float(-v_ref[1])))
        v_from_beta_sin, _, _, mode = resolve_attitude_to_vhat(alpha_t, beta_s, "beta_sin")

        self.assertEqual(mode, "beta_sin")
        np.testing.assert_allclose(v_from_beta_sin, v_ref, rtol=0.0, atol=1e-12)

    def test_resolve_attitude_rejects_non_canonical_attitude_input(self):
        _, _, _, mode_default = resolve_attitude_to_vhat(0.0, 0.0, "")
        self.assertEqual(mode_default, "beta_tan")
        with self.assertRaises(ValueError):
            resolve_attitude_to_vhat(0.0, 0.0, "βsin定義")
        with self.assertRaises(ValueError):
            resolve_attitude_to_vhat(0.0, 0.0, "unknown_mode")


if __name__ == "__main__":
    unittest.main()
