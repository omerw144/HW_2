from tqdm import tqdm
import numpy as np
from interface import Regressor
from utils import Config, get_data
import pandas as pd


class AlsMatrixFactorization(Regressor):
    def __init__(self, config):
        self.k = config.k
        self.gamma_bu = config.gamma_bu
        self.gamma_bi = config.gamma_bi
        self.gamma_x = config.gamma_x
        self.gamma_y = config.gamma_y
        self.train_epochs = config.epochs
        self.n_users = None
        self.n_items = None
        self.user_biases = None  # b_u (users) vector
        self.item_biases = None  # # b_i (items) vector
        self.current_epoch = 0
        self.global_bias = None
        self.x_u = None
        self.y_i = None
        self.val_rmse = [float('inf'), float('inf'), float('inf')]

    def record(self, covn_dict):
        epoch = "{:02d}".format(self.current_epoch)
        temp = f"| epoch   # {epoch} :"
        for key, value in covn_dict.items():
            key = f"{key}"
            val = '{:.4}'.format(value)
            result = "{:<32}".format(F"  {key} : {val}")
            temp += result
        print(temp)

    def calc_regularization(self):
        # Calculate the regularization equation result
        value = self.gamma_bu * (sum(np.square(self.user_biases))) + \
                self.gamma_bi * (sum(np.square(self.item_biases))) + \
                self.gamma_x * (np.sum(np.square(self.x_u))) + \
                self.gamma_y * (np.sum(np.square(self.y_i)))

        return value

    def fit(self, X):
        # set the class attributes like in the linear regression baseline file
        self.n_users = len(set(X[:, 0]))
        self.n_items = len(set(X[:, 1]))
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)
        self.global_bias = np.mean(X[:, 2])
        self.x_u = np.array(np.random.rand(self.k, self.n_users))
        self.y_i = np.array(np.random.rand(self.k, self.n_items))

        if not self.train_epochs:
            while ~((round(self.val_rmse[-3], 3) <= round(self.val_rmse[-2], 3)) & (
                    round(self.val_rmse[-2], 3) <= round(self.val_rmse[-1], 3)) & (self.current_epoch > 4)):
                self.run_epoch(X)

                train_mse = np.square(self.calculate_rmse(X))
                self.val_rmse.append(self.calculate_rmse(validation))

                train_objective = train_mse * X.shape[0] + self.calc_regularization()
                epoch_convergence = {"train_objective": train_objective,
                                     "train_mse": train_mse,
                                     "val_rmse": self.val_rmse[-1]}
                self.record(epoch_convergence)
                self.current_epoch += 1

        else:
            while self.current_epoch < self.train_epochs:
                self.run_epoch(X)
                train_rmse = self.calculate_rmse(X)
                val_rmse = self.calculate_rmse(validation)
                # train_mae = self.calculate_mae(X)
                val_mae = self.calculate_mae(validation)
                # train_r_sq = self.calculate_r_squared(X)
                val_r_sq = self.calculate_r_squared(validation)

                train_objective = np.square(train_rmse) * X.shape[0] + self.calc_regularization()
                epoch_convergence = {"train_objective": train_objective,
                                     "train_rmse": train_rmse,
                                     "Val RMSE": val_rmse,
                                     "Val MAE": val_mae,
                                     "Val R-squared": val_r_sq}
                self.record(epoch_convergence)
                self.current_epoch += 1

    def run_epoch(self, data):
        gamma_bu = self.gamma_bu
        gamma_bi = self.gamma_bi
        gamma_x = self.gamma_x
        gamma_y = self.gamma_y

        for user in list(set(data[:, 0])):
            data_user = data[data[:, 0] == user]

            r_u_i = data_user[:, 2]  # rating
            u_b = self.user_biases[user]
            i_b = self.item_biases[list(set(data_user[:, 1]))]
            #     # setting the 2 matrix as objects
            y_i = self.y_i[:, list(set(data_user[:, 1]))]
            a_u_i = r_u_i - self.global_bias - u_b - i_b
            # self.x_u[:, user] = np.matmul(np.linalg.inv(
            #     np.dot(y_i, y_i.T) + gamma_x * (np.identity(y_i.shape[0]))),
            #     np.dot(a_u_i, y_i.T))
            #######
            sigma_D_u = np.sum((a_u_i * y_i), axis=1)
            multiplier_u = np.sum(y_i.T[:, :, None] * y_i.T[:, None], axis=0)  # (d,d)
            multiplier_u_inv = np.linalg.inv(
                multiplier_u + gamma_x * np.identity(multiplier_u.shape[0]))  # (d,d)
            x_u_new = np.dot(multiplier_u_inv, sigma_D_u)  # (d,)
            self.x_u[:, user] = x_u_new
            #######

            #######
            x_u_T_y_i = np.dot(self.x_u[:, user].T, y_i)  # 1*|D_u|
            sigma_D_u = np.sum(r_u_i - self.global_bias - i_b - x_u_T_y_i)
            multiplier_u = 1 / (data_user.shape[0] + gamma_bu)  # V
            b_u_new = sigma_D_u * multiplier_u
            self.user_biases[user] = b_u_new
            #######
            # self.user_biases[user] = 1 / (len(set(data_user[:, 1])) + gamma_bu) * np.sum(
            #     r_u_i - self.global_bias - i_b - np.dot(self.x_u[:, user].T, y_i))

        for item in list(set(data[:, 1])):
            data_item = data[data[:, 1] == item]
            r_u_i = data_item[:, 2]  # rating
            #     # set the biases for items and users
            u_b = self.user_biases[list(set(data_item[:, 0]))]
            i_b = self.item_biases[item]
            #     # setting the 2 matrix as objects
            y_i = self.y_i[:, item]
            x_u = self.x_u[:, list(set(data_item[:, 0]))]
            x_u_T = x_u.T

            a_u_i = r_u_i - self.global_bias - u_b - i_b

            #######
            sigma_D_i = np.sum((r_u_i - self.global_bias - u_b - i_b) * x_u, axis=1)  # (d, )
            multiplier_i = np.sum(x_u_T[:, :, None] * x_u_T[:, None], axis=0)  # (d,d)
            multiplier_i_inv = np.linalg.inv(
                multiplier_i + gamma_y * np.identity(multiplier_i.shape[0]))  # (d,d)
            y_i_new = np.dot(multiplier_i_inv, sigma_D_i)  # (d,)
            self.y_i[:, item] = y_i_new
            #######

            # self.y_i[:, item] = np.matmul(np.linalg.inv(
            #     np.dot(x_u, x_u.T) + gamma_y * (np.identity(x_u.shape[0]))),
            #     np.dot(a_u_i, x_u.T))

            #######
            x_u_T_y_i = np.dot(x_u_T, y_i)  # |D_u|*1
            sigma_D_i = np.sum(r_u_i - self.global_bias - u_b - x_u_T_y_i)
            multiplier_i = 1 / (data_item.shape[0] + gamma_bi)  # V
            b_i_new = sigma_D_i * multiplier_i
            self.item_biases[item] = b_i_new
            #######

            # self.item_biases[item] = 1 / (len(set(data_item[:, 0])) + gamma_bi) * np.sum(
            #     r_u_i - self.global_bias - u_b - np.dot(x_u.T, self.y_i[:, item]))

    def predict_on_pair(self, user, item):
        try:
            value = self.global_bias + self.user_biases[user] + self.item_biases[item] + np.dot(self.x_u[:, user].T,
                                                                                                self.y_i[:, item])


        except IndexError:  # We assume we know that we don't check for new users, only new items
            value = self.global_bias + self.user_biases[user] + np.mean(self.item_biases)
            # value = self.global_bias + np.mean(self.user_biases) + np.mean(self.item_biases)

        return value


if __name__ == '__main__':
    epoch = None
    results = {}
    best_hyper_params = [None,None,None, float('inf')]
    for gamma in tqdm([0.01,0.1]):
        for k in tqdm([10,24]):
            baseline_config = Config(
                gamma_bu=gamma,
                gamma_bi=gamma,
                gamma_x=gamma,
                gamma_y=gamma,
                k=k,
                epochs=epoch)

            if not epoch:
                train, validation, test = get_data()

                baseline_model = AlsMatrixFactorization(baseline_config)
                baseline_model.fit(train)
                # val_rmse = baseline_model.calculate_rmse(validation)

                test_epoch = baseline_model.current_epoch - 2

                # else:
                #     train, validation = get_data()
                baseline_config = Config(
                    gamma_bu=gamma,
                    gamma_bi=gamma,
                    gamma_x=gamma,
                    gamma_y=gamma,
                    k=k,
                    epochs=test_epoch)

                train = pd.DataFrame(train)
                validation = pd.DataFrame(validation)
                data = pd.concat([train, validation])
                baseline_model = AlsMatrixFactorization(baseline_config)
                baseline_model.fit(np.array(data))
                data_rmse = baseline_model.calculate_rmse(data)
                print(data_rmse)
                if gamma not in results:
                    results[gamma] = {}
                results[gamma][k] = data_rmse

                if data_rmse < best_hyper_params[3]:
                    best_hyper_params = [gamma, k, test_epoch, data_rmse]

    print(results)

    gamma, k, test_epoch = best_hyper_params[:3]
    baseline_config = Config(
        gamma_bu=gamma,
        gamma_bi=gamma,
        gamma_x=gamma,
        gamma_y=gamma,
        k=k,
        epochs=test_epoch)


    ratings = []
    baseline_model = AlsMatrixFactorization(baseline_config)
    baseline_model.fit(np.array(data))
    for row in range(len(test)):
        ratings.append(baseline_model.predict_on_pair(test[row][0], test[row][1]))

    test=pd.DataFrame(test,columns=['User_ID_Alias','Movie_ID_Alias'])
    test['Rating'] = ratings
    test.to_csv("data/B_311337356_203349857.csv")
