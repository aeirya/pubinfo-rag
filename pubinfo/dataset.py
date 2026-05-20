from pathlib import Path
import pandas as pd
import json

DATA_DIR=Path('./data/publications')

DEFAULT_COLUMNS = ['title', 'keywords', 'authors']
ABSTRACTS_AND_KEYWORDS = ['title', 'keywords', 'abstracts']

def list_names() -> list[str]:
    return sorted(p.stem for p in DATA_DIR.glob('*.csv'))

def load_csv(name: str):
    return pd.read_csv(DATA_DIR / f'{name}.csv')

def load_jsonl(name: str):
    text = (DATA_DIR / f'{name}.jsonl').read_text()
    jsons = [json.loads(line.strip()) for line in text.strip().splitlines()]
    return pd.DataFrame(jsons)

def load(name: str, columns=DEFAULT_COLUMNS, limit=None) -> pd.DataFrame:
    main = load_csv(name)
    more = load_jsonl(name)
    df = main.merge(more, on="submission_id", how="left")

    if columns:
        df = df[columns]
    if limit and limit > 0:
        df = df.head(limit)
        
    return df
