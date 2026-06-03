from pandas import DataFrame

from pubinfo.retrieval.config import (
    canonical_kind,
    resolve_columns,
)

from .cache import build_cached_bm25, build_cached_dense
from .impl import bm25, dense


def build_retriever(
    df: DataFrame,
    kind: str = "hybrid",
    columns: str | list[str] | None = "default",
    k: int = 10,
    bm25_k: int | None = None,
    dense_k: int | None = None,
    rrf_k: int | None = None,
    dense_model: str = "BAAI/bge-base-en-v1.5",
    rebuild: bool = False,
    use_cache: bool = True,
):
    kind = canonical_kind(kind)

    corpus = columns if isinstance(columns, str) else "custom"
    resolved_columns = resolve_columns(columns)

    if not use_cache:
        if kind == "bm25":
            return bm25.build(df, k=k, columns=resolved_columns)

        if kind == "dense":
            return dense.build(
                df,
                k=k,
                columns=resolved_columns,
                model_name=dense_model,
            )

        if kind == "hybrid":
            bm = bm25.build(
                df,
                k=bm25_k or k,
                columns=resolved_columns,
            )
            de = dense.build(
                df,
                k=dense_k or k,
                columns=resolved_columns,
                model_name=dense_model,
            )
            return hybrid.merge(
                bm,
                de,
                top_k=k,
                rrf_k=rrf_k or 3 * k,
            )

    if kind == "bm25":
        return build_cached_bm25(
            df=df,
            corpus=corpus,
            columns=resolved_columns,
            k=k,
            rebuild=rebuild,
        )

    if kind == "dense":
        return build_cached_dense(
            df=df,
            corpus=corpus,
            columns=resolved_columns,
            k=k,
            model_name=dense_model,
            rebuild=rebuild,
        )

    if kind == "hybrid":
        bm = build_cached_bm25(
            df=df,
            corpus=corpus,
            columns=resolved_columns,
            k=bm25_k or k,
            rebuild=rebuild,
        )

        de = build_cached_dense(
            df=df,
            corpus=corpus,
            columns=resolved_columns,
            k=dense_k or k,
            model_name=dense_model,
            rebuild=rebuild,
        )

        return hybrid.merge(
            bm,
            de,
            top_k=k,
            rrf_k=rrf_k or 3 * k,
        )

    raise ValueError(f"Unknown retriever kind: {kind!r}")