import numpy as np

def user_based(ratings, is_rated, element_index, similarity_function, neighborhood_selection):
    ratings, is_rated = _remove_rows_without_rating(ratings, is_rated, element_index)
    all_similarities = _all_similarities_with_rows(ratings, is_rated, element_index[0], similarity_function)

    neighborhood_selection_function = neighborhood_selection["function"]
    neighborhood_selection_args = neighborhood_selection.copy()
    del neighborhood_selection_args["function"]
    neighborhood_indices = neighborhood_selection_function(all_similarities, neighborhood_selection_args)

    sum_of_weighted_ratings = sum(
        map(
            lambda row_index: ratings[row_index, element_index[1]] * all_similarities[row_index],
            neighborhood_indices
        )
    )

    sum_of_similarities = sum(
        all_similarities[neighborhood_indices]
    )

    # import pdb; pdb.set_trace()
    return sum_of_weighted_ratings / sum_of_similarities

def item_based(ratings, is_rated, element_index, similarity_function, neighborhood_selection):
    return user_based(ratings.T, is_rated.T, (element_index[1], element_index[0]), similarity_function, neighborhood_selection)

def _all_similarities_with_rows(ratings, is_rated, row_index, similarity_function):
    """Similarities with all other rows with row given by row_index
    Args:
        ratings (numpy.array): userxitem matrix
        is_rated (numpy.array): userxitem boolean matrix
        row_index (int): row index to compute the similarity against
        similarity_function (function): similarity function from recommender.neighborhood.similarity
    Returns:
        one dimensional array of the similarities with the other rows

    The ratings matrix is considered to only have those rows(users), which have a rating for the desired item.
    The desired row must be included as well.
    """
    row_count = ratings.shape[0]
    similarities = np.empty((row_count))
    similarities[row_index] = np.nan
    for x in range(row_count):#all rows
        if x == row_index:
            continue
        similarities[x] = similarity_function(ratings, is_rated, row_index, x)

    return similarities

def _remove_rows_without_rating(ratings, is_rated, element_index):
    is_rated_or_wants_to = is_rated.copy()
    #don't remove the user, the prediction is wanted for
    is_rated_or_wants_to[element_index] = True
    #only those users, who have a rating for the desired item are considered
    return (ratings[is_rated_or_wants_to[:, element_index[1]], :], is_rated[is_rated_or_wants_to[:, element_index[1]],:])
