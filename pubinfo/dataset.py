from pathlib import Path
import pandas as pd

DATA_DIR=Path('./data/publications')
DEFAULT_COLUMNS = ['title', 'keywords', 'authors']

def list_names():
    return sorted(p.stem for p in DATA_DIR.glob('*.csv'))

def load(name: str, columns=DEFAULT_COLUMNS, limit=None):
    df = pd.read_csv(DATA_DIR / f'{name}.csv')
    if columns:
        df = df[columns]
    if limit:
        df = df.head(limit)
        
    return df