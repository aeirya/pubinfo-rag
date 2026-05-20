from pubinfo.retriever import bm25_retriever, faiss_retriever


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


def fuse_retrievers(*retrievers, top_k=10, rrf_k=60):
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

    return fuse_retrievers([bm25, faiss], k, rff_k)
