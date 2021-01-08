import pandas as pd
from config import TRAIN_PATH, VALIDATION_PATH
import numpy as np


def get_data():
    """
    reads train, validation to python indices so we don't need to deal with it in each algorithm.
    of course, we 'learn' the indices (a mapping from the old indices to the new ones) only on the train set.
    if in the validation set there is an index that does not appear in the train set then we can put np.nan or
     other indicator that tells us that.
    """
    train = pd.read_csv("data/Train.csv")
    validation = pd.read_csv("data/Validation.csv")
    test = pd.read_csv("data/Test.csv")
    # print(len(train))
    # print(len(validation))
    # train = train[:(round(len(train) / 100))]
    # validation = validation[:round((len(validation) / 100))]

    movie_list = np.array(pd.concat([train.Movie_ID_Alias, validation.Movie_ID_Alias,test.Movie_ID_Alias]).unique())  # .tolist()
    user_list = np.array(pd.concat([train.User_ID_Alias, validation.User_ID_Alias,test.User_ID_Alias]).unique())  # .tolist()

    # Create arrays of the new IDs
    new_movie_list = np.arange(len(movie_list))
    new_user_list = np.arange(len(user_list))

    # Create dictionaries that map the IDs
    movies_dict = dict(zip(movie_list, new_movie_list))
    users_dict = dict(zip(user_list, new_user_list))

    # Replace the IDs by the dictionaries
    train.replace({"User_ID_Alias": users_dict}, inplace=True)
    train.replace({"Movie_ID_Alias": movies_dict}, inplace=True)

    validation.replace({"User_ID_Alias": users_dict}, inplace=True)
    validation.replace({"Movie_ID_Alias": movies_dict}, inplace=True)

    test.replace({"User_ID_Alias": users_dict}, inplace=True)
    test.replace({"Movie_ID_Alias": movies_dict}, inplace=True)

    return np.array(train), np.array(validation), np.array(test)


class Config:
    def __init__(self, **kwargs):
        self._set_attributes(kwargs)

    def _set_attributes(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
