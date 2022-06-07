import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

def to_pickle(input_path, output_path):
    data = pd.read_csv(input_path)
    data.to_pickle(output_path)

def transform_columns(input_path, output_path):
    data = pd.read_pickle(input_path)
    data.drop(['Id', 'groupId', 'matchId', 'matchType'], axis=1, inplace=True)
    data.dropna(inplace=True)
    data.to_pickle(output_path)

def train_val_split(input_path, train_path, val_path, random_state=42):
    data = pd.read_pickle(input_path)
    train_data, val_data = train_test_split(data, random_state=random_state)
    train_data.to_pickle(train_path)
    val_data.to_pickle(val_path)

def sampling(input_path, output_path, n, random_state=42):
    data = pd.read_pickle(input_path)
    data = data.sample(n=100000, random_state=random_state)
    data.to_pickle(output_path)