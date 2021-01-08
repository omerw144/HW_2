import numpy as np
from numpy import sqrt, square, array


class Regressor:
    def __init__(self):
        raise NotImplementedError

    def fit(self, train):
        raise NotImplementedError

    def predict_on_pair(self, user, item) -> float:
        """given a user and an item predicts the ranking"""

    def calculate_rmse(self, data):
        e = 0
        data = array(data)
        for row in data:
            # index, user, item, rating = row     # run through cvs!!
            user, item, rating = row  # run through get_data!!
            e += square(rating - self.predict_on_pair(user, item))
        return sqrt(e / data.shape[0])

    def calculate_mae(self, data):
        e = 0
        data = array(data)
        for row in data:
            # index, user, item, rating = row     # run through cvs!!
            user, item, rating = row  # run through get_data!!
            e += abs(rating - self.predict_on_pair(user, item))
        return e / data.shape[0]

    def calculate_r_squared(self, data):
        data = array(data)
        preds=[]
        for row in data:
            # index, user, item, rating = row     # run through cvs!!
            user, item, rating = row  # run through get_data!!
            preds.append(self.predict_on_pair(user, item))
        correlation_matrix = np.corrcoef(data[:,2], preds)
        correlation_xy = correlation_matrix[0, 1]
        r_squared = correlation_xy ** 2

        return r_squared
