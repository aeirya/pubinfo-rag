from pandas import DataFrame
from pathlib import Path
import json
from pandas import DataFrame


def stringify(dataset: DataFrame):
    data = dataset.apply(lambda x: str(x), axis=1).to_list()
    data = '\n\n'.join(data)
    return data

def load_jsonl(path: Path|str, to_df=True):
    path = Path(path)
    text = path.read_text()
    jsons = [json.loads(line.strip()) for line in text.strip().splitlines()]
    if to_df:
        return DataFrame(jsons)
    return jsons
