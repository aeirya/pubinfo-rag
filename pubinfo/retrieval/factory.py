from pubinfo.retrieval import build_hybrid
from pandas import DataFrame
from pubinfo.dataset.publication import default_columns, default_columns_no_abstract

COLUMN_PRESETS = {
    "default": default_columns,
    "no_abstract": default_columns_no_abstract,
}

def build_retriever(
    df: DataFrame, 
    k = 10,
    columns:str|list = None,
    bm_cols = default_columns,
    dense_cols = default_columns,
    ):

    if isinstance(columns, str) and columns != '':
        columns = COLUMN_PRESETS[columns]
    if columns is not None:
        bm_cols = dense_cols = columns
    
    return build_hybrid(
        df, 
        k=k, bm25_k=k, faiss_k=k, 
        rrf_k=3*k, 
        bm25_cols=bm_cols, faiss_cols=dense_cols)