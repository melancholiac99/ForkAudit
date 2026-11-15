import unittest
import numpy as np
from exploratory_shop_audit.audit_statistics import reliability_components, reliability, funnel


class StatisticsTests(unittest.TestCase):
    def test_all_ordered_pairs(self):
        y = np.random.default_rng(19).normal(size=(3, 12))
        z = y - y.mean(axis=0)
        expected = np.mean([z[a, k] * z[a, l] for a in range(3) for k in range(12) for l in range(12) if k != l])
        s, n = reliability_components(y)
        self.assertAlmostEqual(s, expected)
        self.assertAlmostEqual(n, np.mean(np.var(z, axis=1, ddof=1)))

    def test_deterministic_signal(self):
        c = reliability_components(np.tile([[0.0], [1.0]], (1, 12)))
        self.assertEqual(c, (.25, 0.0))
        self.assertEqual(reliability([c])['rho']['3'], 1.0)

    def test_negative_signal_is_preserved(self):
        c = reliability_components([[0, 1] * 6, [1, 0] * 6])
        self.assertLess(c[0], 0)
        self.assertFalse(reliability([c])['identifiable'])

    def test_signal_plus_independent_noise(self):
        rng = np.random.default_rng(47)
        components = [reliability_components(np.array([-.5, .5])[:, None] + rng.normal(size=(2, 12))) for _ in range(10000)]
        result = reliability(components)
        self.assertAlmostEqual(result['S'], .25, delta=.02)
        self.assertAlmostEqual(result['N'], .5, delta=.02)
        self.assertAlmostEqual(result['rho']['3'], .6, delta=.03)

    def test_dual_funnel_negative_rewards(self):
        self.assertAlmostEqual(sum(funnel(1, .55, .25, -.85)), 1.85)
        self.assertAlmostEqual(sum(funnel(1, 1, .55, .14)), .86)
        with self.assertRaises(ValueError):
            funnel(.2, .5, .5, .5)


if __name__ == '__main__':
    unittest.main()
