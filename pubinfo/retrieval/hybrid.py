from pubinfo.retrieval import build_bm25, build_dense


def rrf_merge(rankings, rrf_k=60):
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
    ]


def merge(*retrievers, top_k=10, rrf_k=60):
    def retrieve(query):
        rankings = [retriever(query) for retriever in retrievers]
        return rrf_merge(rankings, rrf_k=rrf_k)[:top_k]
    
    return retrieve

def build(
    df,
    k=5,
    bm25_k=20,
    faiss_k=20,
    bm25_cols = None,
    faiss_cols = ['keywords'],
    rff_k = 60,
):

    bm25 = build_bm25(
        df=df,
        k=bm25_k,
        columns=bm25_cols,
    )

    faiss = build_dense(
        df=df,
        k=faiss_k,
        columns=faiss_cols,
    )

    return merge([bm25, faiss], k, rff_k)
