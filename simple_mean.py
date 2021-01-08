from interface import Regressor
from utils import get_data
import pandas as pd
from config import USER_COL_NAME_IN_DATAEST, ITEM_COL_NAME_IN_DATASET, RATING_COL_NAME_IN_DATASET


class SimpleMean(Regressor):
    def __init__(self):
        self.user_means = {}

    def fit(self, X):
        self.user_means = train[['Ratings_Rating', 'User_ID_Alias']].groupby('User_ID_Alias').mean().to_dict()[
            'Ratings_Rating']
        return self.user_means

    def predict_on_pair(self, user: int, item: int):
        return self.user_means[user]


if __name__ == '__main__':
    train, validation = get_data()
    train = pd.DataFrame(train, columns=[USER_COL_NAME_IN_DATAEST, ITEM_COL_NAME_IN_DATASET, RATING_COL_NAME_IN_DATASET])
    validation = pd.DataFrame(validation, columns=[USER_COL_NAME_IN_DATAEST, ITEM_COL_NAME_IN_DATASET, RATING_COL_NAME_IN_DATASET])
    baseline_model = SimpleMean()
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))
