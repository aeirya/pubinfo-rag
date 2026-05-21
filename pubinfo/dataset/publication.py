from pathlib import Path
from .load import read_csv, read_jsonl
from .clean import clean_publication_data
from enum import Enum

data_dir=Path('./data/publications')

class Columns(Enum):
    TITL_KW_ABST_AUTHR_JNL = ['title', 'keywords', 'abstract', 'authors', 'context_name']
    DEFAULT=TITL_KW_ABST_AUTHR_JNL


def list_names() -> list[str]:
    return sorted(p.stem for p in data_dir.glob('*.csv'))

def load_publication_db(name: str):
    csv = read_csv(name)
    jsons = read_jsonl(name)
    df = csv.merge(
        jsons,
        on='submission_id',
        suffixes=(None, '__')
        )
    return df[[c for c in df if not c.endswith('__')]]

def load_db(name: str, columns=Columns.DEFAULT, limit=None):
    df = load_publication_db(name)
    if columns:
        df = df[columns]
    if limit:
        df = df.head(limit)
    return clean_publication_data(df)

