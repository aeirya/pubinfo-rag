from __future__ import annotations
from typing import Callable
from pandas import DataFrame, Series
import pandas as pd
import re
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def __row_to_text__(row: Series, columns=None):
    """
    Convert one dataframe row to searchable text.
    Keeps column names, which helps BM25.
    """
    if columns is None:
        columns = list(row.index)

    return "\n".join(
        f"{col}: {row[col]}"
        for col in columns
        if pd.notna(row[col])
    )

def __make_row_documents__(ds: DataFrame):
    # todo: add metadata
    return [
        Document(
            page_content=str(row), 
            metadata={"row_id": row_id}
            )
        for row_id,row in ds.iterrows()
    ]

def __chunk_documents__(documents: list[Document], chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    return chunks

def __tokenize_and_lower_text__(text):
    return re.findall(r"\w+", text.lower())

def row_retriever(
        ds: DataFrame, 
        k=5, 
        chunk_size=40, 
        chunk_overlap=20,
        bm25_params={"k1": 1.5, "b": 0.85},
        ) -> Callable[[str], DataFrame]:

    docs = __make_row_documents__(ds)
    docs = __chunk_documents__(docs, chunk_size, chunk_overlap)
    
    retriever = BM25Retriever.from_documents(
        docs,
        k=k,
        bm25_params=bm25_params,
        preprocess_func=__tokenize_and_lower_text__,
        )

    def retrieve(query):
        hits = retriever.invoke(query)
        row_ids = list(dict.fromkeys(doc.metadata["row_id"] for doc in hits))
        return ds.loc[row_ids]

    return retrieve

'''
set default retriever behavior
'''
bm25 = row_retriever