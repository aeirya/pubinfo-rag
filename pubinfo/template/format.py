import pandas as pd
from pubinfo.typing import Question


def record_to_text(record: dict, cols=None):
    if cols:
        record = {k:v for k,v in record.items() if k in cols}
    
    return '\n'.join(
        f"{col}: {value}"
        for col, value in record.items()
    )
    
def row_to_text(row: pd.Series, cols=None):
    if cols is None:
        cols = row.index
    return record_to_text(row.to_dict(), cols=cols)

def add_index(index: int, text: str):
    text = '\n'.join('\t'+line for line in text.split('\n'))
    return f'[{index}]' + text
    
def rows_to_text(df: pd.DataFrame, cols=None):
    if cols is None:
        cols = list(df)
    
    records = df.to_dict("records")
    documents = [
        add_index(i, record_to_text(r, cols=cols))
        for i,r in enumerate(records, start=1)
    ]
    return '\n\n'.join(documents)

# def format(o: object):
#     if isinstance(o, pd.DataFrame):
#         return rows_to_text(o)
#     if isinstance(o, pd.Series):
#         return row_to_text(o)
#     if isinstance(o, dict):
#         return record_to_text(o)
#     return repr(o)

def rows_to_context(df: pd.DataFrame, ids: list[int], columns=None):
   return rows_to_text(df.loc[ids], cols=columns)
     
def format_mcq(question: Question):
    ''' multichoice questions '''
    prompt = {x: question[x] for x in 'ABCD'}
    choices = '\n'.join(f'{k}. {v}' for k,v in prompt.items())
    return '\n'.join(
        [question['Question'], choices]
    )
