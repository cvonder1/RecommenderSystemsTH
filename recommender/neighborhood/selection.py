import numpy as np
import math

def top_k(similarities, k):
    """Top k similarities
    Args:
        similarities (numpy.array): one dimensional array
        k (int): number of maximal selected elements(inclusive)
    Returns:
        List of indices for the k highest values in similarities
    """
    try:
        k = args["k"]
    except KeyError:
        raise ValueError("no k in parameters found")
    #Pair<index, value>
    top_k = [(-1, -math.inf) for i in range(k)]
    for i in range(similarities.shape[0]):
        similarity = similarities[i]
        if math.isnan(similarity):
            continue
        min_index = np.argmin(list(map(#in numpy notation: top_k[:,1]
            lambda pair: pair[1],
            top_k
        )))
        #update, if greater value is found
        if math.isinf(top_k[min_index][1]) or similarity > top_k[min_index][1]:
            top_k[min_index] = [i, similarity]

    return np.array(list(
        map(
            lambda pair: pair[0],
            filter(
                lambda pair: not math.isinf(pair[1]),
                top_k
            )
        )
    ))
