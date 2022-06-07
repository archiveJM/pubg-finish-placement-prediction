import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

class RandomForestModel:
    def __init__(self, regr_kargs={}):
        self.model = RandomForestRegressor(**regr_kargs)

    def train(self, /, data=None, data_path=None):
        if data_path is not None:
            data = pd.read_pickle(data_path)
        if data is None:
            raise ValueError('No Data')
        
        X = data.drop('winPlacePerc', axis=1)
        y = data['winPlacePerc']

        self.model.fit(X, y)

        pred = self.model.predict(X)
        print(f'Train MAE: {mean_absolute_error(y, pred)}')

    def validation(self, /, data=None, data_path=None):
        if data_path is not None:
            data = pd.read_pickle(data_path)
        if data is None:
            raise ValueError('No Data')
        
        X = data.drop('winPlacePerc', axis=1)
        y = data['winPlacePerc']

        pred = self.model.predict(X)
        print(f'Val MAE: {mean_absolute_error(y, pred)}')
    
    def load_model(self, model_path):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

    def save_model(self, model_path):
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)