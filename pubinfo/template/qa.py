from pandas import DataFrame

def __record_to_text__(record: dict, i: int):
    rows = [
        f"{col}: {value}"
        for col, value in record.items()
    ]
    rows = [rows[0]] + ['\t' + r for r in rows[1:]]
    text = f'[{i}]\t' + '\n'.join(rows)
    return text


def __to_text__(df: DataFrame):
    records = df.to_dict("records")
    documents = [
        __record_to_text__(r,i) 
        for i,r in enumerate(records, start=1)
    ]
    return '\n\n'.join(documents)


def format(df: DataFrame):
    '''
    formats a DataFrame view as a question. e.g.:

    '''
    return __to_text__(df)