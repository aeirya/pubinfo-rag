from langchain_core.documents import Document
from pandas import Series
import pandas as pd
from pandas import DataFrame

def doc_ids(docs: list[Document]):
    return [doc.metadata["row_id"] for doc in docs]

def build_df_retriver(retriever, df: pd.DataFrame):
    def retrieve(query):
        hits = retriever.invoke(query)
        row_ids = doc_ids(hits)
        return df.loc[row_ids]
    return retrieve

###
# used by the retrievers

def row_to_text(row: Series, columns=None, include_column_names=True):
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
    
def make_documents(df: DataFrame, columns=None, row_fn=None):
    if row_fn is None:
        row_fn = lambda row: row_to_text(row, columns=columns)
        
    return [
        Document(
            page_content=row_fn(row),
            metadata={'row_id': idx}
            ) 
        for idx, row in df.iterrows()
        ]
