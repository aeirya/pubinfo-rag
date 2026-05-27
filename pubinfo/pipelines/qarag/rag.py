from pubinfo.retrieval import Retriever
from pubinfo.typing import Model

from pubinfo.retrieval import SearchResult

class RAG:
    def __init__(self, retriever: Retriever, generate: Model):
        self.retriever = retriever
        self.generate = generate

    def __search(self, query: str) -> SearchResult:
        return self.retriever.search(query)

    def __answer(self, query: str, result: SearchResult):
        return self.generate(
            query=query,
            documents=result.context,
        )
        
    def __call__(self, query: str) -> dict:
        result = self.__search(query)
        answer = self.__answer(query, result)
        return {
            "answer": answer,
            "retrieved_ids": result.ids,
        }
