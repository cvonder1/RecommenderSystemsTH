import numpy as np
import input.filesystem as filesystem
import umf.training as training
import umf.prediction as prediction
import random
from math import sqrt

PATH = "C:\\Users\\vonderschen\\Programming\\thKoeln\\MA2\\shopSystem\\resources\\ratings.csv"

ratings, max = filesystem.read_normalized_rating_matrix(PATH)
is_rated = ratings != 0
ratings_training = ratings.copy()
removed_items = []
#selecting removal items
# TODO: Geht das iterieren auch schöner?
for row_index in range(ratings.shape[0]):
    for column_index in range(ratings.shape[1]):
        if not is_rated[row_index, column_index]:
            continue
        if random.random() <= 0.9: #select ~90% for training
            continue
        removed_items.append((row_index, column_index))
        ratings_training[row_index, column_index] = 0

u, v = training.train(ratings_training, ratings != 0, 1000)

print(u)
print("================")
print(v)
print("================")
prediction_matrix = prediction.predict_all(u, v) * max
print((u @ v.T) * max)

for x in (prediction_matrix == (u @ v.T) * max).flat:
    if not x:
        print("False")

# simple measure for accurancy
total_error = 0
for index in removed_items:
    total_error += abs((ratings[index] * 5 - prediction_matrix[index]))



print("Shape of rating matrix: ", ratings.shape)
print("Size of rating matrix: ", ratings.size)
print("Number of removed ratings: ", len(removed_items))
print("Scale: 1-", max)
print("Middle deviation: ", total_error / len(removed_items))
