"""Unit tests for the geometry/classification core (no camera or GUI needed)."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ksl_transcriber.hand_tracking import build_vector, classify, extract_features


def lm(x, y, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def make_hand(base_x=0.5, base_y=0.5, spread=0.05):
    """A synthetic 21-point 'hand' fanned out from a wrist at (base_x, base_y)."""
    points = [lm(base_x, base_y)]  # wrist
    for i in range(1, 21):
        points.append(lm(base_x + spread * (i % 5), base_y - spread * (i // 5)))
    return points


class TestFeatureExtraction(unittest.TestCase):
    def test_translation_invariance(self):
        hand_a = make_hand(base_x=0.5, base_y=0.5)
        hand_b = make_hand(base_x=0.2, base_y=0.8)  # same shape, different position
        feat_a = extract_features(hand_a)
        feat_b = extract_features(hand_b)
        for a, b in zip(feat_a, feat_b):
            self.assertAlmostEqual(a, b, places=6)

    def test_scale_invariance(self):
        hand_a = make_hand(spread=0.05)
        hand_b = make_hand(spread=0.10)  # same shape, twice the size
        feat_a = extract_features(hand_a)
        feat_b = extract_features(hand_b)
        for a, b in zip(feat_a, feat_b):
            self.assertAlmostEqual(a, b, places=6)

    def test_vector_length(self):
        self.assertEqual(len(extract_features(make_hand())), 42)


class TestBuildVector(unittest.TestCase):
    def test_no_hands(self):
        vec, count = build_vector([], [])
        self.assertIsNone(vec)
        self.assertEqual(count, 0)

    def test_one_hand_zero_pads_second_half(self):
        vec, count = build_vector([make_hand()], ['Left'])
        self.assertEqual(count, 1)
        self.assertEqual(len(vec), 84)
        self.assertTrue(all(v == 0.0 for v in vec[42:]))

    def test_two_hands_deterministic_order(self):
        left = make_hand(base_x=0.2)
        right = make_hand(base_x=0.8)
        # Feed them in reverse order; sorting by label should make it deterministic.
        vec1, count1 = build_vector([right, left], ['Right', 'Left'])
        vec2, count2 = build_vector([left, right], ['Left', 'Right'])
        self.assertEqual(count1, count2, 2)
        self.assertEqual(vec1, vec2)


class TestClassify(unittest.TestCase):
    def setUp(self):
        base = [0.0] * 84
        a_vec = list(base)
        a_vec[0] = 1.0
        b_vec = list(base)
        b_vec[0] = -1.0
        self.samples = [
            {'label': 'A', 'hands': 1, 'vector': a_vec},
            {'label': 'A', 'hands': 1, 'vector': [v + 0.01 for v in a_vec]},
            {'label': 'B', 'hands': 1, 'vector': b_vec},
        ]

    def test_matches_closest_label(self):
        query = [0.0] * 84
        query[0] = 0.98
        result = classify(query, 1, ['A', 'B'], self.samples)
        self.assertIsNotNone(result)
        self.assertEqual(result['label'], 'A')
        self.assertGreater(result['confidence'], 0.5)

    def test_no_samples_for_hand_count_returns_none(self):
        query = [0.0] * 84
        result = classify(query, 2, ['A', 'B'], self.samples)
        self.assertIsNone(result)

    def test_far_from_everything_returns_none(self):
        query = [10.0] * 84
        result = classify(query, 1, ['A', 'B'], self.samples)
        self.assertIsNone(result)

    def test_candidate_label_filter(self):
        # B is the true nearest neighbor overall; restricting candidates to
        # ['A'] should make classify() ignore B and match A instead.
        base = [0.0] * 84
        a1, a2, b1 = list(base), list(base), list(base)
        a1[0], a2[0], b1[0] = 1.0, 1.01, 0.55
        samples = [
            {'label': 'A', 'hands': 1, 'vector': a1},
            {'label': 'A', 'hands': 1, 'vector': a2},
            {'label': 'B', 'hands': 1, 'vector': b1},
        ]
        query = list(base)
        query[0] = 0.5

        unrestricted = classify(query, 1, ['A', 'B'], samples)
        self.assertEqual(unrestricted['label'], 'B')

        restricted = classify(query, 1, ['A'], samples)
        self.assertEqual(restricted['label'], 'A')


if __name__ == '__main__':
    unittest.main()
