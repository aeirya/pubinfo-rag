from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pandas import DataFrame, Series
from .common import doc_ids, row_to_text

def make_documents(df: DataFrame, columns=None):
    return [
        Document(
            page_content=row_to_text(row, columns=columns),
            metadata={'row_id': idx}
            ) 
        for idx, row in df.iterrows()
        ]

def build(
        df: DataFrame, 
        k=4, 
        columns=None,
        model_name="BAAI/bge-base-en-v1.5",
        ):
    
    docs = make_documents(df, columns)
    emb = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.from_documents(docs, emb)

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={'k': k}
    )
    return lambda query: doc_ids(retriever.invoke(query))