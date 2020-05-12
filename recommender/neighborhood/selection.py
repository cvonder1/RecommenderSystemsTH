import numpy as np

def top_k(similarities, k):
    """Top k similarities
    Args:
        similarities (numpy.array): one dimensional array
        k (int): number of maximal selected elements(inclusive)
    Returns:
        List of indices for the k highest values in similarities
    """
    #for very big datasets this "algorithm" is not ideal
    return np.argsort(similarities)[-(k+1):]
