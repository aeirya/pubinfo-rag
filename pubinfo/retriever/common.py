from langchain_core.documents import Document

def doc_ids(docs: list[Document]):
    return [doc.metadata["row_id"] for doc in docs]

def init(retriever, ds):
    def retrieve(query):
        hits = retriever.invoke(query)
        row_ids = doc_ids(hits)
        return ds.loc[row_ids]
    return retrieve