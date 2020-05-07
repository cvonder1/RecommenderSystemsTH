from numpy import genfromtxt, uint8, float64
import numpy as np

def read_rating_matrix(path, delimiter=","):
    return genfromtxt(path, delimiter=delimiter, dtype=uint8)

"""
:param int max: the maximal value in matrix
:return: matrix normalized to values between 0 and 1 and max value of matrix
:rtype: (numpy.array, int)
"""
def normalize_values(matrix, max=None):
    if max == None:
        max = np.max(matrix)
    return (matrix.astype(float64) / max, max)

def read_normalized_rating_matrix(path, max=None):
    return normalize_values(read_rating_matrix(path), max)
