from pubinfo.retriever import bm25, faiss

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


def hybrid_retriever(
    df,
    k=5,
    bm25_k=20,
    faiss_k=20,
    columns=None,
):
    bm25 = bm25.retriever(
        df=df,
        k=bm25_k,
        columns=columns,
    )

    faiss = faiss.retriever(
        df=df,
        k=faiss_k,
        columns=columns,
    )

    def retrieve(query):
        bm25_ids = bm25(query)
        faiss_ids = faiss(query)

        return rrf_merge([bm25_ids, faiss_ids], k=k)

    return retrieve