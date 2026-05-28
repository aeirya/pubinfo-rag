from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pandas import DataFrame
from .documents import doc_ids, make_documents


def build(
        df: DataFrame, 
        k=4, 
        columns=None,
        model_name="BAAI/bge-base-en-v1.5",
        ):
    
    docs = make_documents(df, columns)
    emb = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})
    db = FAISS.from_documents(docs, emb)

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={'k': k}
    )
    return lambda query: doc_ids(retriever.invoke(query))
