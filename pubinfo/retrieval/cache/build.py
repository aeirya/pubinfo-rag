# factory.py

from pandas import DataFrame

from ..impl import bm25, dense
from . import io as cache


def build_cached_bm25(
    df: DataFrame,
    corpus: str,
    columns,
    k: int,
    rebuild: bool = False,
):
    path = cache.path_for(corpus, "bm25", "index.pkl")

    index = cache.cached(
        path=path,
        load=lambda: cache.load_pickle(path),
        build=lambda: bm25.build_index(df, columns),
        save=lambda obj: cache.save_pickle(path, obj),
        rebuild=rebuild,
    )

    return bm25.as_retriever(index, k=k)


def build_cached_dense(
    df: DataFrame,
    corpus: str,
    columns,
    k: int,
    model_name: str,
    rebuild: bool = False,
):
    emb = dense.get_embeddings(model_name)
    path = cache.path_for(corpus, "dense", model_name)

    index = cache.cached(
        path=path,
        load=lambda: cache.load_faiss(path, emb),
        build=lambda: dense.build_index(df, columns, model_name),
        save=lambda obj: cache.save_faiss(path, obj),
        rebuild=rebuild,
    )

    return dense.as_retriever(index, k=k)