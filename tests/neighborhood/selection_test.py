import numpy as np
import unittest

import recommender.neighborhood.selection as selection

class SelectionTest(unittest.TestCase):

    def test_k_top(self):
        similarities = np.array([0.5, 0.8, -0.1, 0.3, 0.9])

        top_k = selection.top_k(similarities, 3)

        assert 0 in top_k
        assert 1 in top_k
        assert 4 in top_k

    def test_k_top_for_less_elements_than_k(self):
        similarities = np.array([-0.4, 1])

        top_k = selection.top_k(similarities, 3)

        assert 0 in top_k
        assert 1 in top_k
