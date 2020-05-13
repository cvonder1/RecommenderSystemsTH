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

import pytest

class ItemBasedPredictionTest(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def setup_mocks(self, mocker):
        self.user_based_mock = mocker.patch("recommender.neighborhood.prediction.user_based")

    def test_item_based_prediction_for_alice_and_item5(self):
        ratings = np.array([
            [5, 4, 0],
            [3, 2, 3],
            [4, 4, 5],
            [3, 1, 4],
            [1, 5, 1],
        ])

        is_rated = ratings != 0

        self.user_based_mock.return_value = 0.8

        prediction_alice_item5 = prediction.item_based(
            ratings,
            is_rated,
            element_index=(0,2),
            similarity_function=similarity.cosine_similarity,
            neighborhood_selection={
                "function": selection.top_k,
                "k": 3
            }
        )

        assert prediction_alice_item5 == 0.8

        self.user_based_mock.assert_called_once()
        call = self.user_based_mock.call_args[0]
        assert (call[0] == ratings.T).all()
        assert (call[1] == is_rated.T).all()
        assert call[2] == (2, 0)
        assert call[3] == similarity.cosine_similarity
        assert call[4] == {
            "function": selection.top_k,
            "k": 3
        }


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
