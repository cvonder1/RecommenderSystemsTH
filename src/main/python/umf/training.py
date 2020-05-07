import numpy as np

# The error matrix has the same shape as the ratings matrix

LEARNING_RATE = 0.005 # alpha

def train(ratings, is_rated, k_factors):
    """Trains a model with k_factors latent vectors based on the ratings matrix.

    Args:
        ratings (numpy.array): userxitem rating matrix
        is_rated (numpy.array): boolean matrix in shape of ratings matrix, specifieng which ratings are present
        k_factors (int): number of latent vectors

    Returns:
        (numpy.array, numpy.array) U and V matrix::

            U and V are factors of R, so that R ~= U.dot(V.T) is true

    values of userxitem matrix should be in range [0;1].
    Migth work with other scales, but I am not sure about that
    """
    user_factors = init_factors_matrix(ratings.shape[0], k_factors) #U
    item_factors = init_factors_matrix(ratings.shape[1], k_factors) #V

    current_error_matrix = np.ones(ratings.shape, dtype=np.float64)
    previous_error_matrix = current_error_matrix

    while True:
        previous_user_factors = user_factors.copy()
        previous_item_factors = item_factors.copy()

        previous_error_matrix = current_error_matrix
        current_error_matrix = error_matrix(ratings, is_rated, user_factors, item_factors)
        print("Total error: ", total_error(current_error_matrix))

        user_factors = previous_user_factors + LEARNING_RATE * (current_error_matrix @ previous_item_factors)
        item_factors = previous_item_factors + LEARNING_RATE * (current_error_matrix.T @ previous_user_factors)

        if is_convergenced(previous_error_matrix, current_error_matrix):
            break

    return (user_factors, item_factors)

def error_matrix(ratings, is_rated, user_factors, item_factors):
    #TODO: actually only present ratings must be considered when computing error_matrix
    error_matrix = ratings - (user_factors @ item_factors.T)
    error_matrix[is_rated == False] = 0 #all not rated items cannot be considered in error matrix
    return error_matrix

import math

def init_factors_matrix(count, k_factors):
    return np.random.rand(count, k_factors) / k_factors#dividing, so that the matrix is nearer to a optimal solution

def is_convergenced(previous_error_matrix, current_error_matrix):
    if abs(total_error(previous_error_matrix) - total_error(current_error_matrix)) < 8:
        return True
    return False

def total_error(error_matrix): #sigma J
    squared_sum = 0.
    for e in error_matrix.flat:
        squared_sum += 0.5 * (e ** 2)
    return squared_sum
