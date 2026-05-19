from pathlib import Path
import pandas as pd

DATA_DIR=Path('./data/publications')
DEFAULT_COLUMNS = ['title', 'keywords', 'authors']

def list_names() -> list[str]:
    return sorted(p.stem for p in DATA_DIR.glob('*.csv'))

def load(name: str, columns=DEFAULT_COLUMNS, limit=None) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f'{name}.csv')
    if columns:
        df = df[columns]
    if limit and limit > 0:
        df = df.head(limit)
        
    return df
