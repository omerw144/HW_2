import numpy as np
from scipy.spatial.distance import pdist, squareform
from utils import get_data, Config
from interface import Regressor
from os import path
from config import CORRELATION_PARAMS_FILE_PATH_ZIP


class KnnItemSimilarity(Regressor):
    def __init__(self, config):
        self.n_users = None
        self.n_items = None
        self.corr_dict = None
        self.k = config.k
        self.similarity_values_weighted_sum = {}
        self.similarity_values = {}
        self.sim_mat = []
        self.train = []
        self.validation = []
        self.counter = 0

    def correlation(self, v, u):
        """Calculate pearson Similarity between v and u vectors \
        create u and v as binary vectors, where indices are not none in both of them"""
        u_bin = u[~(np.isnan(v) | np.isnan(u))]
        v_bin = v[~(np.isnan(v) | np.isnan(u))]

        if (len(u_bin) == 0) | (len(v_bin) == 0):
            corr = 0
        else:
            corr = np.corrcoef(u_bin, v_bin)[0, 1]

            if corr < 0:
                corr = 0

        self.counter += 1

        return corr

    def calculate_similarity(self, X):
        # function for calculate distances according to our correlation function
        pdi = pdist(X, self.correlation)
        # creating the array of correlation between users
        sim_mat = squareform(pdi)
        # to make sure there is no NaN in our array
        sim_mat = np.nan_to_num(sim_mat, 0)
        return sim_mat

    def predict_on_pair(self, user, item):
        """Function to calc the prediction estimation, using the formula from the assignment\""
        ""Returns prediction for the rating of the given item and user"""
        # return the average rating of that user item if the item was not rated in train set by any user.
        k = self.k
        data = self.train
        sim_mat = self.sim_mat
        if item not in self.train.index:
            prediction = np.nanmean(self.train[user])

            return prediction

        # list of items that are not null for the user
        items_by_user = list(self.train[user][self.train[user].notnull()].index)

        # finding the proper value for each user items match
        distance = sim_mat[item][items_by_user]

        # sorted list of the k most similar items
        zipped_items_sim = sorted(list(zip(items_by_user, distance)), key=lambda x: x[1], reverse=True)[:k]

        # Calculating the formula for estimation according to the formula in our HW doc
        down = sum([x[1] for x in zipped_items_sim])
        up = sum(data[user][[d[0] for d in zipped_items_sim]] * [x[1] for x in zipped_items_sim])

        if (up == 0 and down == 0) or np.isnan(up / down):
            # In case all items that user rated are 0 similarity with the item:
            # In this case we predict the mean rating of that item if it was rated in train set.
            prediction = np.nanmean(
                data.loc[item])
            return prediction
        prediction = up / down
        return prediction

    def fit(self, train_set, test_set):
        """For all the ratings in test_set, calc the differance between real rating and predicted one,\
        in order to compute the MAE measure"""
        self.train = train_set
        self.validation = test_set
        if path.isfile(CORRELATION_PARAMS_FILE_PATH_ZIP):
            self.sim_mat = self.upload_params()
        else:
            self.sim_mat = self.calculate_similarity(self.train)
            self.save_params()

    def upload_params(self):
        corr_params = np.array(pd.read_csv(CORRELATION_PARAMS_FILE_PATH_ZIP, index_col=0))
        return corr_params

    def save_params(self):
        pd.DataFrame(self.sim_mat).to_csv(CORRELATION_PARAMS_FILE_PATH_ZIP, compression='zip')


if __name__ == '__main__':
    import pandas as pd

    knn_config = Config(k=25)
    train, validation = get_data()
    train_df = pd.DataFrame({'users': train[:, 0], 'movies': train[:, 1], 'rating': train[:, 2]}) \
        .pivot(index='users', columns='movies', values='rating').T
    validation_df = pd.DataFrame({'users': validation[:, 0], 'movies': validation[:, 1], 'rating': validation[:, 2]}) \
        .pivot(index='users', columns='movies', values='rating').fillna(0).T

    knn = KnnItemSimilarity(knn_config)
    knn.fit(train_set=train_df, test_set=validation_df)
    print(knn.calculate_rmse(validation))
