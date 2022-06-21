from typing import List
import os
from torch.utils.data import Dataset
import app

def load_data(fn:str, fp=None):
    '''
    V1 06.19.2022,
    :param fn: local file name
    :param fp: full path for dataset
    :return: map of { text : emotion }
    '''
    if fp:
        txt_path = fp
    else:
        txt_path = os.path.join(app.root(), "datasets", "emotions", fn)

    res = {}
    with open(txt_path) as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.split(";")
            line, emotion = line_split
            emotion = emotion.strip()
            if emotion == 'joy' or emotion == 'anger':
                res[line] = emotion.strip()
    print(f'loaded {len(res)} items')
    return res

def load_data_externally(fp: str):
    return load_data(None, fp)

def one_hot_encode_values_map_fn(data_list: List[str]) -> dict[str, float]:
    '''
    Input: data_list
    :param data_list: list of strings for labels to encode between 0 and 1.
    for example ['joy', 'anger']
    :return:
    map of { value: float }
    for example
    '''
    n = len(data_list)
    encoded_values_to_float = {}
    for i, val in enumerate(data_list):
        encoded_values_to_float[val] = i / (n - 1)
    print(encoded_values_to_float)
    return encoded_values_to_float

def one_hot_encode_values(data_map: dict[str, str or float], values_map: dict[str, float]) -> dict[str, float]:
    '''

    :param data_map: original data containing { string : string }
    for example { 'This is a sentence', 'joy' }
    :param values_map: Map of values to one h   ot encoded float
    for example { 'joy' : 0. }
    :return: data map is modified in and returns { string : float }
    for example { 'This is a sentence' : 0. }
    '''
    for k, v in data_map.items():
        data_map[k] = values_map[v]
    return data_map

class HappyClassifierDataset(Dataset):
    def __init__(self):
        train_data_map = load_data("train.txt")
        values_map = one_hot_encode_values_map_fn(['joy', 'anger'])
        one_hot_encode_values(train_data_map, values_map)
        self.train_data = list(train_data_map.items())

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx]