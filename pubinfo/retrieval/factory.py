from pandas import DataFrame
from pubinfo.retrieval.bm25 import build as build_bm25
from pubinfo.retrieval.config import (
    RetrievalConfig,
    canonical_kind,
    resolve_columns,
)
from pubinfo.retrieval.dense import build as build_dense
from pubinfo.retrieval.hybrid import build as build_hybrid

def build_retriever(
    df: DataFrame, 
    k=10,
    columns: str | list[str] | None = None,
    bm_cols: str | list[str] | None = None,
    dense_cols: str | list[str] | None = None,
    kind: str = "hybrid",
    bm25_k: int | None = None,
    dense_k: int | None = None,
    rrf_k: int | None = None,
    dense_model: str = "BAAI/bge-base-en-v1.5",
):
    config = RetrievalConfig(
        kind=kind,
        k=k,
        columns=columns,
        bm25_k=bm25_k,
        dense_k=dense_k,
        rrf_k=rrf_k,
        bm25_columns=bm_cols,
        dense_columns=dense_cols,
        dense_model=dense_model,
    )
    return build_retriever_from_config(df, config)


def build_retriever_from_config(df: DataFrame, config: RetrievalConfig):
    kind = canonical_kind(config.kind)
    columns = resolve_columns(config.columns)
    bm_cols = resolve_columns(config.bm25_columns)
    dense_cols = resolve_columns(config.dense_columns)

    if columns is not None:
        bm_cols = dense_cols = columns

    if kind == "bm25":
        return build_bm25(df, k=config.k, columns=bm_cols)

    if kind == "dense":
        return build_dense(
            df,
            k=config.k,
            columns=dense_cols,
            model_name=config.dense_model,
        )

    if kind != "hybrid":
        raise ValueError(f"Unknown retriever kind: {config.kind!r}")

    return build_hybrid(
        df, 
        k=config.k,
        bm25_k=config.bm25_k or config.k,
        faiss_k=config.dense_k or config.k,
        rrf_k=config.rrf_k or 3 * config.k,
        bm25_cols=bm_cols,
        faiss_cols=dense_cols,
    )
