import unittest

import torch

from .post_d_gradients import geometry


class AdjacentGradients(unittest.TestCase):
    def test_same_opposite_and_zero(self):
        a = {'u':torch.tensor([3., 0.]), 'v':torch.tensor([-4.])}
        opposite = {key:-2 * v for key, v in a.items()}
        row = geometry(a, opposite)
        self.assertEqual((row['cosine'], row['left_norm'], row['right_norm']), (-1, 5, 10))
        self.assertEqual(geometry(a, a)['cosine'], 1)
        self.assertIsNone(geometry(a, {k:v * 0 for k,v in a.items()})['cosine'])

    def test_mismatched_coordinates(self):
        with self.assertRaises(ValueError):
            geometry({'a':torch.ones(1)}, {'b':torch.ones(1)})
