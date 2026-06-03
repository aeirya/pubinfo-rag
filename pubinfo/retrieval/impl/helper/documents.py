import pandas as pd
from langchain_core.documents import Document
from pandas import DataFrame, Series


def doc_ids(docs: list[Document]):
    return [doc.metadata["row_id"] for doc in docs]


def row_to_text(row: Series, columns=None, include_column_names=True):
    if columns is None:
        columns = list(row.index)

    if not include_column_names:
        return "\t".join(str(value) for value in row[columns])

    return "\n".join(
        f"{column}: {row[column]}"
        for column in columns
        if column in row and pd.notna(row[column])
    )


def make_documents(df: DataFrame, columns=None, row_fn=None):
    if row_fn is None:
        row_fn = lambda row: row_to_text(row, columns=columns)

    return [
        Document(
            page_content=row_fn(row),
            metadata={"row_id": idx},
        )
        for idx, row in df.iterrows()
    ]
