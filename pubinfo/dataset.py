from pathlib import Path
import pandas as pd

DATA_DIR=Path('./data/publications')

def list_names():
    return set([p.stem for p in DATA_DIR.glob('*')])

def load(name: str, columns=['title', 'keywords', 'authors']):
    df = pd.read_csv(DATA_DIR / f'{name}.csv')
    if columns:
        df = df[columns]
    return df