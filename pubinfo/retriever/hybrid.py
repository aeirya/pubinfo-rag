from pubinfo.retriever import bm25_retriever, faiss_retriever


def rrf_merge(rankings, k=10, rrf_k=60):
    """
    Reciprocal Rank Fusion
    It merges ranked lists without needing scores to be comparable

    rankings: list of ranked row_id lists
    returns: fused row_ids
    """
    scores = {}

    for ranking in rankings:
        for rank, row_id in enumerate(ranking):
            scores[row_id] = scores.get(row_id, 0) + 1 / (rrf_k + rank + 1)

    return [
        row_id
        for row_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ][:k]


def fuse(bm25, faiss, k):
    def retrieve(query):
        bm25_ids = bm25(query)
        faiss_ids = faiss(query)

        return rrf_merge([bm25_ids, faiss_ids], k=k)
    
    return retrieve

def retriever(
    df,
    k=5,
    bm25_k=20,
    faiss_k=20,
    bm25_cols = None,
    faiss_cols = ['keywords']
):

    bm25 = bm25_retriever(
        df=df,
        k=bm25_k,
        columns=bm25_cols,
    )

    faiss = faiss_retriever(
        df=df,
        k=faiss_k,
        columns=faiss_cols,
    )

    return fuse
