from pandas import DataFrame
from pathlib import Path
import json
from pandas import DataFrame
from pubinfo.typing import Question

def stringify(dataset: DataFrame):
    data = dataset.apply(lambda x: str(x), axis=1).to_list()
    data = '\n\n'.join(data)
    return data

def format_dict(d: dict):
    return '\n\n'.join([f'{k}: {v}' for k,v in d.items()])

def formatted_print(d: dict):
    text = format_dict(d)
    for line in text.split('\n'):
        print(line)

def format_question(question: Question):
    prompt = {x: question[x] for x in 'ABCD'}
    choices = '\n'.join(f'{k}. {v}' for k,v in prompt.items())
    return '\n'.join(
        [question['Question'], choices]
    )

def load_jsonl(path: Path|str, to_df=True):
    path = Path(path)
    text = path.read_text()
    jsons = [json.loads(line.strip()) for line in text.strip().splitlines()]
    if to_df:
        return DataFrame(jsons)
    return jsons
