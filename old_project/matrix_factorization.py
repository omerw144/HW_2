from tqdm import tqdm
import numpy as np
from old_project.interface import Regressor
from old_project.utils import Config, get_data
import pandas as pd


class MatrixFactorization(Regressor):
    def __init__(self, config):
        self.lr_bu = config.lr_bu
        self.lr_bi = config.lr_bi
        self.lr_x = config.lr_x
        self.lr_y = config.lr_y
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
            while ~((round(self.val_rmse[-3], 4) <= round(self.val_rmse[-2], 4)) & (
                    round(self.val_rmse[-2], 4) <= round(self.val_rmse[-1], 4)) & (self.current_epoch > 4)):
                self.run_epoch(X)

                train_mse = np.square(self.calculate_rmse(X))
                self.val_rmse.append(self.calculate_rmse(validation))

                train_objective = train_mse * X.shape[0] + self.calc_regularization()
                epoch_convergence = {"train_objective": train_objective,
                                     "train_mse": train_mse,
                                     "val_rmse": self.val_rmse[-1]}
                self.record(epoch_convergence)
                self.current_epoch += 1
                self.lr_bu *= 0.9
                self.lr_bi *= 0.9
                self.lr_x *= 0.9
                self.lr_y *= 0.9

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
                self.lr_bu *= 0.9
                self.lr_bi *= 0.9
                self.lr_x *= 0.9
                self.lr_y *= 0.9

    def run_epoch(self, data):
        gamma_bu = self.gamma_bu
        gamma_bi = self.gamma_bi
        gamma_x = self.gamma_x
        gamma_y = self.gamma_y
        lr_bu = self.lr_bu
        lr_bi = self.lr_bi
        lr_x = self.lr_x
        lr_y = self.lr_y
        mu = self.global_bias

        for i in data:
            r_u_i = i[2]  # rating
            user = i[0]  # user_id
            item = i[1]  # item id
            # set the biases for items and users
            u_b = self.user_biases[i[0]]
            i_b = self.item_biases[i[1]]
            # setting the 2 matrix as objects
            x_u = self.x_u
            y_i = self.y_i

            # calculate the error of an instance
            e_ui = r_u_i - mu - u_b - i_b - np.dot(x_u[:, user].T, y_i[:, item])

            # derivatives calculation for each of the estimators
            derivative_1 = -2 * (e_ui - gamma_bu * u_b)
            derivative_2 = -2 * (e_ui - gamma_bi * i_b)

            derivative_3 = -2 * (e_ui * x_u[:, user] - gamma_y * y_i[:, item])
            derivative_4 = -2 * (e_ui * y_i[:, item] - gamma_x * x_u[:, user])

            # update user and item biases
            self.user_biases[i[0]] -= derivative_1 * lr_bu
            self.item_biases[i[1]] -= derivative_2 * lr_bi
            # update x and y matrices
            self.y_i[:, item] -= derivative_3 * lr_y
            self.x_u[:, user] -= derivative_4 * lr_x

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
    best_hyper_params = [None, None, None,None, float('inf')]
    for gamma in tqdm([0.01, 0.1]):
        for k in tqdm([10, 24]):
            for lr in tqdm([0.01, 0.05]):
                baseline_config = Config(
                    lr_bu=lr,
                    lr_bi=lr,
                    lr_x=lr,
                    lr_y=lr,
                    gamma_bu=gamma,
                    gamma_bi=gamma,
                    gamma_x=gamma,
                    gamma_y=gamma,
                    k=k,
                    epochs=epoch)

                if not epoch:
                    train, validation,test = get_data()
                    baseline_model = MatrixFactorization(baseline_config)
                    baseline_model.fit(train)
                    # val_rmse = baseline_model.calculate_rmse(validation)
                    test_epoch = baseline_model.current_epoch - 2

                    # else:
                    #     train, validation = get_data()

                    baseline_config = Config(
                        lr_bu=lr,
                        lr_bi=lr,
                        lr_x=lr,
                        lr_y=lr,
                        gamma_bu=gamma,
                        gamma_bi=gamma,
                        gamma_x=gamma,
                        gamma_y=gamma,
                        k=k,
                        epochs=test_epoch)
                    train = pd.DataFrame(train)
                    validation = pd.DataFrame(validation)
                    data = pd.concat([train, validation])
                    baseline_model = MatrixFactorization(baseline_config)
                    baseline_model.fit(np.array(data))
                    data_rmse = baseline_model.calculate_rmse(data)
                    print(data_rmse)

                    if gamma not in results:
                        results[gamma] = {}
                    if k not in results[gamma]:
                        results[gamma][k] = {}
                    results[gamma][k][lr] = data_rmse

                    if data_rmse < best_hyper_params[4]:
                        best_hyper_params = [gamma, k, lr, test_epoch, data_rmse]
    print(results)

    gamma, k, lr, test_epoch = best_hyper_params[:4]
    baseline_config = Config(
        lr_bu=lr,
        lr_bi=lr,
        lr_x=lr,
        lr_y=lr,
        gamma_bu=gamma,
        gamma_bi=gamma,
        gamma_x=gamma,
        gamma_y=gamma,
        k=k,
        epochs=test_epoch)


    ratings = []
    baseline_model = MatrixFactorization(baseline_config)
    baseline_model.fit(np.array(data))
    for row in range(len(test)):
        ratings.append(baseline_model.predict_on_pair(test[row][0], test[row][1]))

    test=pd.DataFrame(test,columns=['User_ID_Alias','Movie_ID_Alias'])
    test['Rating'] = ratings
    test.to_csv("data/A_311337356_203349857.csv")