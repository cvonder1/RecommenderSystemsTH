import numpy as np
import math

"""
All computations are row based. Thus a userxitem matrix will lead to a user-based similarity.
If a item-based approach is desired, transpose the matrix.
"""

def cosine_similarity(ratings, is_rated, first_row, second_row):
    """Cosine Similarity between the two rows in the ratings matrix specified by first_row and second_row
    Args:
        ratings (numpy.array): userxitem ratings matrix
        is_rated (numpy.array): userxitem boolean matrix, True -> user rated item
        first_row (int): index of user row
        second_row (int): index of item column

    Returns:
        The raw cosine similarity between the two rows.

        For item-based similarity just transpose the ratings and is_rated matrices.
    """
    return _similarity_between_rows(ratings[[first_row, second_row]], is_rated[[first_row, second_row]], average_function=lambda x,y,z: 0)

def adjusted_cosine_similarity(ratings, is_rated, first_row, second_row):
    """Adjusted Cosine Similarity between the two rows in the ratings matrix specified by first_row and second_row
    Args:
        ratings (numpy.array): userxitem ratings matrix
        is_rated (numpy.array): userxitem boolean matrix, True -> user rated item
        first_row (int): index of user row
        second_row (int): index of item column

    Returns:
        The adjusted cosine similarity between the two rows.

        For item-based similarity just transpose the ratings and is_rated matrices.
    """
    return _similarity_between_rows(ratings[[first_row, second_row]], is_rated[[first_row, second_row]], average_function=_average_of_both_rated)

def pearson_correlation(ratings, is_rated, first_row, second_row):
    """Pearson Correlation between the two rows in the ratings matrix specified by first_row and second_row
    Args:
        ratings (numpy.array): userxitem ratings matrix
        is_rated (numpy.array): userxitem boolean matrix, True -> user rated item
        first_row (int): index of user row
        second_row (int): index of item column

    Returns:
        The Pearson Correlation between the two rows.

        For item-based similarity just transpose the ratings and is_rated matrices.
    """
    return _similarity_between_rows(ratings[[first_row, second_row]], is_rated[[first_row, second_row]], average_function=_average_of_all_rated)

def _similarity_between_rows(ratings, is_rated, average_function):

    average_first_row = average_function(ratings, is_rated, 0)
    average_second_row = average_function(ratings, is_rated, 1)

    #only consider items rated by both aka. k \element I_u \intersect I_v
    ratings_both_rated = ratings[:, _both_rated(is_rated)]

    sum_of_rating_products = _sum_of_mean_centered_rating_products(ratings_both_rated, average_first_row, average_second_row)

    sum_of_centered_first_row_ratings = _sum_of_mean_centered_ratings(ratings_both_rated[0], average_first_row)

    sum_of_centered_second_row_ratings = _sum_of_mean_centered_ratings(ratings_both_rated[1], average_second_row)

    return sum_of_rating_products / (sum_of_centered_first_row_ratings * sum_of_centered_second_row_ratings)


def _both_rated(is_rated):
    return is_rated[0] * is_rated[1]

def _sum_of_mean_centered_rating_products(ratings_both_rated, average_first_row, average_second_row):
    #from Recommender Systems: The Textbook, p. 35
    return sum(#sum over k \element I_u \intersect I_v
        map(
            #(r_u,k - r_u_average) * (r_v,k - r_v_average)
            lambda pair: _mean_center(pair[0], average_first_row) * _mean_center(pair[1], average_second_row),
            ratings_both_rated.T
        )
    )

def _sum_of_mean_centered_ratings(ratings, average):
    #sqrt(sum over k I_u \intersect I_v ((r_uk - r_u_average) ^ 2)))
    #from Recommender Systems: The Textbook, p. 35
    return math.sqrt(
        sum(
            map(
                lambda rating: _mean_center(rating, average) ** 2,
                ratings
            )
        )
    )

def _mean_center(rating, average):
    return rating - average

def _average_of_both_rated(ratings, is_rated, row):
    both_rated = _both_rated(is_rated)
    return np.average(ratings[row, both_rated])

def _average_of_all_rated(ratings, is_rated, row):
    average = np.average(ratings[(row, is_rated[row])])

    if math.isnan(average):
        return 0
    else:
        return average
