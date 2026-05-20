from pubinfo.retrieval import build_hybrid
from pandas import DataFrame

def build_retriever(
    df: DataFrame, 
    k = 10,
    bm_cols = ['title', 'abstract', 'keywords', 'context_name'],
    dense_cols = ['title', 'keywords', 'context_name']
    ):
    return build_hybrid(
        df, 
        k=k, bm25_k=k, faiss_k=k, 
        rrf_k=3*k, 
        bm25_cols=bm_cols, faiss_cols=dense_cols)