from typing import Dict

import numpy as np
import pandas as pd

from interface import Regressor
from utils import Config, get_data
import pickle

from config import BASELINE_PARAMS_FILE_PATH


class Baseline(Regressor):
    def __init__(self, config):
        self.lr = config.lr
        self.gamma = config.gamma
        self.train_epochs = config.epochs
        self.n_users = None
        self.n_items = None
        self.user_biases = None  # b_u (users) vector
        self.item_biases = None  # # b_i (items) vector
        self.current_epoch = 0
        self.global_bias = None

    def record(self, covn_dict: Dict):
        epoch = "{:02d}".format(self.current_epoch)
        temp = f"| epoch   # {epoch} :"
        for key, value in covn_dict.items():
            key = f"{key}"
            val = '{:.4}'.format(value)
            result = "{:<32}".format(F"  {key} : {val}")
            temp += result
        print(temp)

    def calc_regularization(self):
        """Calculating the regularization equation """
        value = self.gamma * (sum(np.square(self.user_biases)) + sum(np.square(self.item_biases)))
        return value

    def fit(self, X):
        # number of users
        self.n_users = len(set(X[:, 0]))
        # number of items
        self.n_items = len(set(X[:, 1]))
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)
        # global bias set as mean
        self.global_bias = np.mean(X[:, 2])
        while self.current_epoch < self.train_epochs:
            self.run_epoch(X)
            train_mse = np.square(self.calculate_rmse(X))
            train_objective = train_mse * X.shape[0] + self.calc_regularization()
            epoch_convergence = {"train_objective": train_objective,
                                 "train_mse": train_mse}
            self.record(epoch_convergence)
            self.current_epoch += 1
        self.save_params()

    def run_epoch(self, data: np.array):
        gamma = self.gamma
        lr = self.lr
        mu = self.global_bias

        for i in data:
            # actual rating
            r_u_i = i[2]
            # user and item biases
            u_b = self.user_biases[i[0]]
            i_b = self.item_biases[i[1]]
            # the derivative of each of the biases estimators
            derivative_1 = -2 * (r_u_i - mu - u_b - i_b) + gamma * u_b
            derivative_2 = -2 * (r_u_i - mu - u_b - i_b) + gamma * i_b
            # updating user biases and item biases based on the epoch run
            self.user_biases[i[0]] -= derivative_1 * lr
            self.item_biases[i[1]] -= derivative_2 * lr

    def predict_on_pair(self, user: int, item: int):
        try:
            value = self.global_bias + self.user_biases[user] + self.item_biases[item]
        except IndexError:
            # the prediction for the user that is only in the validation set
            value = self.global_bias + self.user_biases[user] + np.mean(self.item_biases)
        return value

    def save_params(self):
        # saving the params as pickle
        dic_params = {'global': self.global_bias, 'users': self.user_biases, 'items': self.item_biases}
        pickle_out = open(BASELINE_PARAMS_FILE_PATH, "wb")
        pickle.dump(dic_params, pickle_out)
        pickle_out.close()


if __name__ == '__main__':
    baseline_config = Config(
        lr=0.001,
        gamma=0.001,
        epochs=10)

    train, validation = get_data()
    baseline_model = Baseline(baseline_config)
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))
