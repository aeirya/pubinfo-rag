from pathlib import Path
import pandas as pd
from pubinfo import util


DATA_DIR=Path('./data/publications')

DEFAULT_COLUMNS = ['title', 'keywords', 'authors']
ABSTRACTS_AND_KEYWORDS = ['title', 'keywords', 'abstract']

def list_names() -> list[str]:
    return sorted(p.stem for p in DATA_DIR.glob('*.csv'))

def load_csv(name: str):
    return pd.read_csv(DATA_DIR / f'{name}.csv')

def load_jsonl(name: str):
    return util.load_jsonl(DATA_DIR / f'{name}.jsonl')

def load_db(name: str, columns=DEFAULT_COLUMNS, limit=None) -> pd.DataFrame:
    main = load_csv(name)
    more = load_jsonl(name)
    df = main.merge(more, on="submission_id", how="outer", suffixes=(None,'_copy'))

    if columns:
        df = df[columns]
    if limit and limit > 0:
        df = df.head(limit)
        
    return df
