# bm25.py

import re

from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .helper.documents import doc_ids, make_documents


def chunk_documents(documents, chunk_size=80, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def tokenize_and_lower(text):
    return re.findall(r"\w+", text.lower())


def df_row_ids(docs):
    ids = doc_ids(docs)
    return list(dict.fromkeys(ids))


def build_index(
    df,
    columns=None,
    chunk_size=80,
    chunk_overlap=20,
    bm25_params=None,
):
    bm25_params = bm25_params or {"k1": 1.5, "b": 0.85}

    docs = make_documents(df, columns)
    chunks = chunk_documents(docs, chunk_size, chunk_overlap)

    return BM25Retriever.from_documents(
        chunks,
        bm25_params=bm25_params,
        preprocess_func=tokenize_and_lower,
    )


def as_retriever(index, k=5):
    index.k = k
    return lambda query: df_row_ids(index.invoke(query))


def build(df, k=5, columns=None):
    index = build_index(df, columns)
    return as_retriever(index, k)