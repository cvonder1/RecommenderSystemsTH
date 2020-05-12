import numpy as np
import unittest

import recommender.neighborhood.selection as selection

class SelectionTest(unittest.TestCase):

    def test_k_top(self):
        similarities = np.array([0.5, 0.8, -0.1, 0.3, 0.9])


        assert 0 in top_k
        assert 1 in top_k
        assert 4 in top_k

    def test_k_top_for_less_elements_than_k(self):
        similarities = np.array([-0.4, 1])

        top_k = selection.top_k(similarities, 3)

        assert 0 in top_k
        assert 1 in top_k
    def test_top_k_for_nan(self):
        similarities = np.array([np.nan, 0.99624059, 0.99388373, 0.93834312, 0.76570486])

        top_k = selection.top_k(similarities, {"k": 3})

        assert 1 in top_k
        assert 2 in top_k
        assert 3 in top_k
