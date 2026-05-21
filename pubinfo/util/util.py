from pandas import DataFrame
from pathlib import Path
from pandas import DataFrame

def stringify(dataset: DataFrame):
    data = dataset.apply(lambda x: str(x), axis=1).to_list()
    data = '\n\n'.join(data)
    return data

def format_dict(d: dict):
    return '\n\n'.join([f'{k}: {v}' for k,v in d.items()])

