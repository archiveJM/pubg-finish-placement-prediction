import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

class RandomForestModel:
    def __init__(self, regr_kargs={}):
        self.model = RandomForestRegressor(**regr_kargs)

    def train(self, train_path):
        data = pd.read_pickle(train_path)
        X = data.drop('winPlacePerc', axis=1)
        y = data['winPlacePerc']
        self.model.fit(X, y)
        print(f'Train MAE: {mean_absolute_error(y, self.predict(X))}')
    
    def load_model(self, model_path):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

    def save_model(self, model_path):
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)