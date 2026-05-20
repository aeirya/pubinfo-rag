from pandas import DataFrame


def record_to_text_with_index(record: dict, i: int):
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
        record_to_text_with_index(r,i) 
        for i,r in enumerate(records, start=1)
    ]
    return '\n\n'.join(documents)


def format(df: DataFrame, cols=None):
    '''
    enumerates dataframe rows
    and stringifies them
    '''
    if cols:
        df = df[cols]
    return __to_text__(df)