import numpy as np

def predict_all(user_factors, item_factors):
    """Predicts a rating matrix based on the U and V matrix given.
    Args:
        user_factors (numpy.array): The U matrix
        item_factors (numpy.array): The V matrix

    Returns:
        The matrix product of user_features and item_features
    """
    return user_factors @ item_factors.T
