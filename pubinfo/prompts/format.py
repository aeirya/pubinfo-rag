import pandas as pd


def record_to_text(record: dict, cols=None):
    if cols:
        record = {key: value for key, value in record.items() if key in cols}

    return "\n".join(
        f"{column}: {value}"
        for column, value in record.items()
    )


def row_to_text(row: pd.Series, cols=None):
    if cols is None:
        cols = row.index
    return record_to_text(row.to_dict(), cols=cols)


def add_index(index: int, text: str):
    text = "\n".join("\t" + line for line in text.split("\n"))
    return f"[{index}]" + text


def rows_to_text(df: pd.DataFrame, cols=None):
    if cols is None:
        cols = list(df)

    documents = [
        add_index(i, record_to_text(record, cols=cols))
        for i, record in enumerate(df.to_dict("records"), start=1)
    ]
    return "\n\n".join(documents)


def rows_to_context(df: pd.DataFrame, ids: list[int], columns=None):
    return rows_to_text(df.loc[ids], cols=columns)


def format_mcq(question: dict[str, str]):
    choices = "\n".join(f"{key}. {question[key]}" for key in "ABCD")
    return "\n".join([question["Question"], choices])
