from pathlib import Path
from .load import read_csv, read_jsonl
from .clean import clean_publication_data

data_dir=Path('./data/publications')

default_columns = ['title', 'keywords', 'abstract', 'authors', 'date_published', 'context_name']
default_columns_no_abstract = list(set(default_columns) - set(['abstract']))

def list_names() -> list[str]:
    return sorted(p.stem for p in data_dir.glob('*.csv'))

def load_publication_db(name: str):
    csv = read_csv(data_dir / name)
    jsons = read_jsonl(data_dir / name)
    df = csv.merge(
        jsons,
        on='submission_id',
        suffixes=(None, '__')
        )
    return df[[c for c in df if not c.endswith('__')]]

def load_db(name: str, columns=default_columns, limit=None):
    df = load_publication_db(name)
    if columns:
        df = df[columns]
    if limit:
        df = df.head(limit)
    return clean_publication_data(df)

