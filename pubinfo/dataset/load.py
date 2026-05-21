import pandas as pd
from pathlib import Path
import json

def read_csv(path: Path):
    df = pd.read_csv(Path(path).with_suffix('.csv'))
    df.columns = df.columns.str.strip()
    return df

def read_jsonl(path: Path|str, to_df=True):
    text = Path(path).with_suffix('.jsonl').read_text().strip()
    jsons = [json.loads(line) for line in text.splitlines()]
    if to_df:
        return pd.DataFrame(jsons)
    return jsons

def load_data(name: str) -> pd.DataFrame:
    path = Path(name)
    # if not path.exists():
    #     paths = list(path.glob(f"./data/*{name}.*"))
    #     if len(paths) == 0:
    #         paths = list(path.glob(f'./data/**/*{name}.*'))
    #     path = Path(paths[0])
    
    if path.suffix == '.csv':
        return read_csv(path)
    if path.suffix == '.jsonl':
        return read_jsonl(path)