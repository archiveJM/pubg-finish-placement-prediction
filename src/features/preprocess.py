import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

def get_data(data):
    """
    data 경로의 파일을 불러와서 df로 반환.
    data 가 str 이 아니면 그대로 반환.
    """
    if isinstance(data, str):
        ext = os.path.splitext(data)[-1]
        if ext == '.pkl':
            data = pd.read_pickle(data)
        elif ext == '.csv':
            data = pd.read_csv(data)
    return data

def save_data(data: 'pd.DataFrame', path:str):
    if path is None:
        return
    ext = os.path.splitext(path)[-1]
    if ext == '.pkl':
        data.to_pickle(path)
    elif ext == '.csv':
        data.to_csv(path)

def dropna(input_data, output_path=None):
    data = get_data(input_data)
    data = data.dropna()
    save_data(data, output_path)
    return data

def add_groupsize(input_data, output_path=None):
    data = get_data(input_data)
    data = data.merge(
        right = data.groupId.value_counts().rename('groupSize'),
        left_on = 'groupId',
        right_index = True,
    )
    save_data(data, output_path)
    return data

def select_numeric(input_data, output_path=None):
    data = get_data(input_data)
    data = data.select_dtypes(include='number')
    save_data(data, output_path)
    return data

def train_val_split(input_data, train_path=None, val_path=None, random_state=42):
    data = get_data(input_data)
    train_data, val_data = train_test_split(data, random_state=random_state)
    save_data(train_data, train_path)
    save_data(val_data, val_path)
    return train_data, val_data

def sampling(input_data, output_path=None, *, n, random_state=42):
    data = get_data(input_data)
    data = data.sample(n=100000, random_state=random_state)
    save_data(data, output_path)
    return data

def split_Xy(input_data, target='winPlacePerc'):
    data = get_data(input_data)
    X = data.drop('winPlacePerc', axis=1)
    y = data['winPlacePerc']
    return X, y