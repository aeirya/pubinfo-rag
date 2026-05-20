import re
from pandas import DataFrame
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .include import doc_ids, make_documents


def chunk_documents(documents: list[Document], chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    return chunks

def tokenize_and_lower(text):
    return re.findall(r"\w+", text.lower())

def df_row_ids(docs: list[Document]):
    ''' return unique row ids while preserving the order of appearence '''
    ids = doc_ids(docs)
    return list(dict.fromkeys(ids))

def build(
        df: DataFrame, 
        k=5, 
        columns=None,
        chunk_size=80, 
        chunk_overlap=20,
        bm25_params={"k1": 1.5, "b": 0.85},
        ):

    docs = make_documents(df, columns)
    docs = chunk_documents(docs, chunk_size, chunk_overlap)
    
    retriever = BM25Retriever.from_documents(
        docs,
        k=k,
        bm25_params=bm25_params,
        preprocess_func=tokenize_and_lower,
        )

    return lambda query: df_row_ids(retriever.invoke(query))
