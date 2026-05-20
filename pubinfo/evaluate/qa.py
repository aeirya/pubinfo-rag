from pandas import DataFrame
from typing import Callable
from pubinfo.template import context
from pubinfo.util import stringify
from pubinfo.util import format_question

LOG_PROMPT = False

Validator = Callable[[str|dict, dict], bool]
Model = Callable[[dict], str|dict]

def get_records(tests: DataFrame):
    return tests.to_dict(orient='records')

def evaluate_qa(db: DataFrame, tests: DataFrame, model: Model, validate: Validator):
    right_ids = []
    wrong_ids = []
    outs = []
    
    for i, test in enumerate(get_records(tests)):
        prompt = format_question(test)
        
        if LOG_PROMPT:
            print('prompt:')
            print(prompt)
        
        out = model(prompt)
        if validate(out, test):
            right_ids += [i]
        else:
            wrong_ids += [i]
            
        outs.append(out)

    R = len(right_ids)
    W = len(wrong_ids)
    return R / (R+W), outs