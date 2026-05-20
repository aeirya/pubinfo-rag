from langchain_core.documents import Document
from pandas import Series

def doc_ids(docs: list[Document]):
    return [doc.metadata["row_id"] for doc in docs]

def init(retriever, ds):
    def retrieve(query):
        hits = retriever.invoke(query)
        row_ids = doc_ids(hits)
        return ds.loc[row_ids]
    return retrieve

def row_to_text(row: Series, columns=None, include_column_names=False):
    """
    Convert one dataframe row to searchable text.
    """
    if columns is None:
        columns = list(row.index)

    if not include_column_names:
        return '\t'.join(str(v) for v in row[columns])

    return "\n".join(
        f"{col}: {row[col]}"
        for col in columns
        if pd.notna(row[col])
    )