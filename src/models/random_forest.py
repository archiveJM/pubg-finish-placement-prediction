import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from ..features.preprocess import split_Xy

class RandomForestModel:
    def __init__(self, regr_kargs={}):
        self.model = RandomForestRegressor(**regr_kargs)

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