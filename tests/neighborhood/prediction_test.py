import unittest
import math
import numpy as np

import recommender.neighborhood.prediction as prediction
import recommender.neighborhood.similarity as similarity
import recommender.neighborhood.selection as selection

class UserBasedPredictionTest(unittest.TestCase):

    def test_user_based_prediction_for_alice_and_item5(self):
        ratings = np.array([
            [5, 4, 0],
            [3, 2, 3],
            [4, 4, 5],
            [3, 1, 4],
            [1, 5, 1],
        ])

        is_rated = ratings != 0

        prediction_alice_item5 = prediction.user_based(
            ratings,
            is_rated,
            element_index=(0,2),
            similarity_function=similarity.cosine_similarity,
            neighborhood_selection= {
                "function": selection.top_k,
                "k": 3
            }
        )

        assert math.isclose(prediction_alice_item5, 3.999195190)

class UtilityTest(unittest.TestCase):

    def test_remove_rows_without_rating(self):
        ratings = np.array([
            [5, 3, 4, 4, 0],
            [3, 1, 2, 3, 3],
            [4, 3, 4, 3, 0],
            [3, 3, 1, 5, 4],
            [1, 5, 5, 2, 0],
        ])

        is_rated = ratings != 0

        cleaned_ratings, cleaned_is_rated = prediction._remove_rows_without_rating(ratings, is_rated, (0, 4))

        assert cleaned_ratings.shape == cleaned_is_rated.shape
        assert cleaned_ratings.shape == (3, 5)

        assert cleaned_ratings[0,2] == 4
        assert cleaned_ratings[1,2] == 2
        assert cleaned_ratings[2,2] == 1
