import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from ..features.preprocess import split_Xy

class Trainer:
    def __init__(self, model=None):
        self.model = model
    
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

class LGBM_Trainer(Trainer):
    def __init__(self, **params):
        self.model = LGBMRegressor(**params)

class RandomForest_Trainer(Trainer):
    def __init__(self, **params):
        self.model = RandomForestRegressor(**params)

class DecisionTree_Trainer(Trainer):
    def __init__(self, **params):
        self.model = DecisionTreeRegressor(**params)

class LinearRegression_Trainer(Trainer):
    def __init__(self, **params):
        self.model = LinearRegression(**params)