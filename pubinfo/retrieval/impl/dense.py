# dense.py

from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from .helper.documents import doc_ids, make_documents


@lru_cache(maxsize=2)
def get_embeddings(model_name: str, device: str = "cpu"):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
    )


def build_index(df, columns=None, model_name="BAAI/bge-base-en-v1.5"):
    docs = make_documents(df, columns)
    emb = get_embeddings(model_name)
    return FAISS.from_documents(docs, emb)


def as_retriever(db, k=4):
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return lambda query: doc_ids(retriever.invoke(query))


def build(df, k=4, columns=None, model_name="BAAI/bge-base-en-v1.5"):
    db = build_index(df, columns, model_name)
    return as_retriever(db, k)