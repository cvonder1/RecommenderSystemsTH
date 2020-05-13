import numpy as np
import unittest
import math

import recommender.neighborhood.similarity as similarity

class AverageTest(unittest.TestCase):

    def test_average_of_all_rated(self):
        ratings = np.array([
            [1, 3, 5],
            [3, 2, 0]
        ])
        is_rated = ratings != 0

        assert similarity._average_of_all_rated(ratings, is_rated, 0) == 3

    def test_average_of_all_rated_zero_for_zero_weights(self):
        ratings = np.zeros((3, 3))
        is_rated = np.array([
            [True, True, True],
            [True, False, True]
        ])

        assert similarity._average_of_all_rated(ratings, is_rated, 0) == 0

    def test_average_of_both_rated(self):
        ratings = np.array([
            [1, 3, 5],
            [3, 2, 0]
        ])
        is_rated = ratings != 0

        assert similarity._average_of_both_rated(ratings, is_rated, 0) == 2

    def test_average_of_both_rated_zero_for_zero_weights(self):
        ratings = np.zeros((3, 3))
        is_rated = np.array([
            [True, True, True],
            [True, False, True]
        ])

        assert similarity._average_of_both_rated(ratings, is_rated, 0) == 0

class SimilarityTest(unittest.TestCase):

    def test_pearson_correlation_for_alice_and_user1(self):
        """Based on Recommender Systems: An Introduction, p. 14"""
        ratings = np.array([
            [5, 3, 4, 4, 0],
            [3, 1, 2, 3, 3],
            [4, 3, 4, 3, 5],
            [3, 3, 1, 5, 4],
            [1, 5, 5, 2, 1],
        ])

        is_rated = ratings != 0

        print(similarity.pearson_correlation(ratings, is_rated, 0, 1))
        assert math.isclose(similarity.pearson_correlation(ratings, is_rated, 0, 1), 0.8391813581)

    def test_cosine_similarity_for_alice_and_user1(self):
        ratings = np.array([
            [5, 3, 4, 4, 0],
            [3, 1, 2, 3, 3],
            [4, 3, 4, 3, 5],
            [3, 3, 1, 5, 4],
            [1, 5, 5, 2, 1],
        ])

        is_rated = ratings != 0

        assert math.isclose(similarity.cosine_similarity(ratings, is_rated, 0, 1), 0.9753213049, abs_tol=0.005)

    def test_adjusted_cosine_similarity_for_alice_and_user1(self):
        ratings = np.array([
            [5, 3, 4, 4, 0],
            [3, 1, 2, 3, 3],
            [4, 3, 4, 3, 5],
            [3, 3, 1, 5, 4],
            [1, 5, 5, 2, 1],
        ])

        is_rated = ratings != 0

        assert math.isclose(similarity.adjusted_cosine_similarity(ratings, is_rated, 0, 1), 0.8528028653, abs_tol=0.005)


class UtilityTest(unittest.TestCase):

    def test_both_rated(self):
        is_rated = np.array([
            [True, False, True],
            [True, True, False]
        ])

        both_rated = similarity._both_rated(is_rated)

        assert both_rated[0] == True
        assert both_rated[1] == False
        assert both_rated[2] == False

    def test_sum_of_mean_centered_rating_products(self):
        ratings_both_rated = np.array([
            [5, 3, 4, 4],
            [3, 1, 2, 3]
        ])
        average_first_row = 4
        average_second_row = 2.4

        assert similarity._sum_of_mean_centered_rating_products(ratings_both_rated, average_first_row, average_second_row) == 2

    def test_sum_of_mean_centered_ratings_for_item1(self):
        ratings = np.array([5, 3, 4, 4])
        average = 4

        assert math.isclose(similarity._sum_of_mean_centered_ratings(ratings, average), 1.414213562)

    def test_sum_of_mean_centered_ratings_for_item2(self):
        ratings = np.array([3, 1, 2, 3])
        average = 2.4

        assert math.isclose(similarity._sum_of_mean_centered_ratings(ratings, average), 1.685229955)
#
# class AdjustedCosineTest(unittest.TestCase):
#
#     def test_adjusted_cosine_similarity_for
