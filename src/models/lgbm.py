import pandas as pd
import pickle
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

from ..features.preprocess import split_Xy

class LGBM_Model:
    def __init__(self, **params):
        self.model = lgb.LGBMRegressor(**params)

    def train(self, data):
        X, y = split_Xy(data)

        self.model.fit(X, y)

        pred = self.model.predict(X)
        print(f'Train MAE: {mean_absolute_error(y, pred)}')

    def validation(self, data):
        X, y = split_Xy(data)

        pred = self.model.predict(X)
        print(f'Val MAE: {mean_absolute_error(y, pred)}')

    def load_model(self, model_path):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

    def save_model(self, model_path):
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)