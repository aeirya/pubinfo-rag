from pathlib import Path
import pandas as pd
from pubinfo.util import util, text


DATA_DIR=Path('./data/publications')

DEFAULT_COLUMNS = ['title', 'keywords', 'authors']
ABSTRACTS_AND_KEYWORDS = ['title', 'keywords', 'abstract']
MORE = list(dict.fromkeys(DEFAULT_COLUMNS + ABSTRACTS_AND_KEYWORDS))

def list_names() -> list[str]:
    return sorted(p.stem for p in DATA_DIR.glob('*.csv'))

def load_csv(name: str):
    return pd.read_csv(DATA_DIR / f'{name}.csv')

def load_jsonl(name: str):
    return util.load_jsonl(DATA_DIR / f'{name}.jsonl')

def clean_data(df: pd.DataFrame):
    if 'abstract' in df:
        df['abstract'] = df['abstract'].apply(text.clean_abstract).apply(text.quote)
    if 'authors' in df:
        df['authors'] = df['authors'].apply(text.clean_authors)
        
    return df

def load_db(name: str, columns=MORE, limit=None) -> pd.DataFrame:
    main = load_csv(name)
    more = load_jsonl(name)
    
    df = main.merge(more, on="submission_id", how="outer", suffixes=(None,'__copy__'))
    df = df[[c for c in df if not c.endswith('__copy__')]]
        
    if columns:
        df = df[columns]
    if limit and limit > 0:
        df = df.head(limit)
    
    df = clean_data(df)
    return df
