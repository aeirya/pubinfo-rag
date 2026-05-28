from typing import Protocol

from pubinfo.retrieval import Retriever

from pubinfo.retrieval import SearchResult


class Generator(Protocol):
    def __call__(self, *, query: str, documents: str) -> str: ...


class RAG:
    def __init__(self, retriever: Retriever, generate: Generator):
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
