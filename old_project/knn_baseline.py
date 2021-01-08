import numpy as np
import pandas as pd
from old_project.interface import Regressor
from old_project.utils import get_data, Config
from old_project.config import BASELINE_PARAMS_FILE_PATH, CORRELATION_PARAMS_FILE_PATH_ZIP


class KnnBaseline(Regressor):
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
        self.global_bias = None
        self.user_biases = None  # b_u (users) vector
        self.item_biases = None  # # b_i (items) vector
        self.counter = 0

    def fit(self, X):
        self.train = X
        self.upload_params()


    def predict_baseline_value(self, user: int, item: int):
        try:
            value = self.global_bias + self.user_biases[user] + self.item_biases[item]
        except IndexError:
            value = self.global_bias + self.user_biases[user] + np.mean(self.item_biases)
        return value

    def predict_on_pair(self, user, item):
        """Function to calc the prediction estimation, using the formula from the assignment\""
        ""Returns prediction for the rating of the given item and user"""
        # return the average rating of that user item if the item was not rated in train set by any user.

        k = self.k
        data = self.train
        sim_mat = self.sim_mat
        value = self.predict_baseline_value(user, item)
        if item not in self.train.index:
            prediction = np.nanmean(self.train[user])
            print('prediction- not in index', prediction)

            return prediction

        # list of items that are not null for the user
        items_by_user = list(self.train[user][self.train[user].notnull()].index)

        # finding the proper value for each user items match
        distance = sim_mat[item][items_by_user]

        # sorted list of the k most similar items
        zipped_items_sim = sorted(list(zip(items_by_user, distance)), key=lambda x: x[1], reverse=True)[:k]

        down = sum([x[1] for x in zipped_items_sim])
        list_1 = [data[user][d[0]] - self.predict_baseline_value(user, d[0]) for d in zipped_items_sim]
        list_2 = [x[1] for x in zipped_items_sim]
        up = np.dot(list_1, list_2)

        if (up == 0 and down == 0) or np.isnan(up / down):
            # In case all items that user rated are 0 similarity with the item:
            # In this case we predict the mean rating of that item if it was rated in train set.
            prediction = np.nanmean(
                data.loc[item])
            return prediction
        prediction = value + (up / down)
        return prediction

    def upload_params(self):
        baseline_params = pd.read_pickle(BASELINE_PARAMS_FILE_PATH)
        self.global_bias = baseline_params['global']
        self.user_biases = baseline_params['users']
        self.item_biases = baseline_params['items']
        self.sim_mat = np.array(pd.read_csv(CORRELATION_PARAMS_FILE_PATH_ZIP, index_col=0))


if __name__ == '__main__':
    baseline_knn_config = Config(k=25)
    train, validation = get_data()
    train_df = pd.DataFrame({'users': train[:, 0], 'movies': train[:, 1], 'rating': train[:, 2]}) \
        .pivot(index='users', columns='movies', values='rating').T
    knn_baseline = KnnBaseline(baseline_knn_config)
    knn_baseline.fit(train_df)

    print(knn_baseline.calculate_rmse(validation))
