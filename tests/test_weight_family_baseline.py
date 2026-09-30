from __future__ import annotations

import unittest

import numpy as np

from scripts.weight_family_baseline import calibrate, parameter_map
from scripts.weight_baselines import make_assignment


class WeightFamilyBaselineTests(unittest.TestCase):
    def test_parameter_map_is_contiguous_and_fixed(self):
        families = parameter_map()
        self.assertEqual(sum(item["count"] for item in families), 2_961_630)
        self.assertEqual(families[0]["start"], 0)
        for left, right in zip(families, families[1:]):
            self.assertEqual(left["stop"], right["start"])

    def test_small_family_falls_back_to_global(self):
        noisy = np.arange(32, dtype=np.float32)
        clean = 2.0 * noisy + 1.0
        labels = np.array([0, 1] * 16, dtype=np.int8)
        family = [{"name": "tiny", "shape": [16], "count": 16, "start": 0, "stop": 16}]
        result = calibrate(noisy, clean, labels, family, min_fitting=64)
        self.assertTrue(result["families"][0]["fallback"])
        self.assertEqual(result["families"][0]["fallback_reason"], "insufficient_fitting_count")

    def test_audit_changes_do_not_change_coefficients_or_family_metrics(self):
        labels = make_assignment(256, half_length=128, seed=2301, block_size=32, guard=0)
        noisy = np.linspace(-2.0, 2.0, 256, dtype=np.float32)
        clean = 1.25 * noisy + 0.4
        families = [{"name": "one", "shape": [128], "count": 128, "start": 0, "stop": 128}]
        first = calibrate(noisy, clean, labels, families, min_fitting=1)
        changed = clean.copy()
        changed[labels == 2] += 10_000.0
        second = calibrate(noisy, changed, labels, families, min_fitting=1)
        self.assertEqual(first["global_coefficients"], second["global_coefficients"])
        self.assertEqual(first["family_affine_development_rmse_original_units"], second["family_affine_development_rmse_original_units"])


if __name__ == "__main__":
    unittest.main()
