from __future__ import annotations

import unittest

import numpy as np

from scripts.weight_baselines import fit_coefficients, make_assignment


class WeightBaselineTests(unittest.TestCase):
    def test_assignment_is_disjoint_and_exhaustive_or_guarded(self):
        labels = make_assignment(512, half_length=256, seed=2301, block_size=32, guard=32)
        self.assertTrue(np.isin(labels, [-1, 0, 1, 2]).all())
        self.assertEqual(labels.shape, (512,))
        self.assertGreater(np.count_nonzero(labels == -1), 0)
        # Every non-guard position has exactly one split code; guards are the only gaps.
        self.assertEqual(np.count_nonzero(np.isin(labels, [0, 1, 2])) + np.count_nonzero(labels == -1), 512)
        for offset in range(256):
            self.assertEqual(labels[offset], labels[256 + offset])

    def test_fit_does_not_use_audit_labels(self):
        labels = make_assignment(512, half_length=256, seed=17, block_size=32, guard=32)
        noisy = np.linspace(-2.0, 2.0, 512)
        clean = 1.25 * noisy + 0.4
        changed = clean.copy()
        changed[labels == 2] += 10000.0
        first = fit_coefficients(noisy, clean, labels)
        second = fit_coefficients(noisy, changed, labels)
        self.assertEqual(first, second)

    def test_fit_does_not_use_development_labels(self):
        labels = make_assignment(512, half_length=256, seed=19, block_size=32, guard=32)
        noisy = np.linspace(-2.0, 2.0, 512)
        clean = 0.75 * noisy - 0.2
        changed = clean.copy()
        changed[labels == 1] += 10000.0
        first = fit_coefficients(noisy, clean, labels)
        second = fit_coefficients(noisy, changed, labels)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
