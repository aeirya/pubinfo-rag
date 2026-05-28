import random

from pubinfo.retrieval import Retriever, SearchResult
from pubinfo.typing import Model


class GRetriever:
    retriever: Retriever
    
    def __init__(self, retriever:Retriever):
        assert retriever is not None

        self.retriever = retriever
        self.k = self.retriever.k
        self.rng = random.Random(42)
        
    def _sample_items(self, items: list[int], n: int):
        if len(items) <= n:
            return items

        keep_indices = set(self.rng.sample(range(len(items)), n))
        return [x for i, x in enumerate(items) if i in keep_indices]
 
    def _insert_randomly(self, inserts: list[int], items: list[int]):
        """Insert each item from inserts at a random position."""
        result = items[:]

        for x in inserts:
            pos = self.rng.randrange(len(result) + 1)
            result.insert(pos, x)

        return result
    
    def ids(self, query: str, gold_ids: list[int]):
        assert len(gold_ids) <= self.k
        
        found = self.retriever.ids(query)
        gold_set = set(gold_ids)
        non_gold = [x for x in found if x not in gold_set]

        n_keep = self.k - len(gold_ids)
        non_gold = self._sample_items(non_gold, n_keep)
        return self._insert_randomly(gold_ids, non_gold)
            
    def search(self, query: str, gold_ids: list[int]) -> SearchResult:
        ids = self.ids(query, gold_ids)
        context = self.retriever.to_context(ids)
        return SearchResult(ids=ids, context=context)
    
    
class GRag:
    def __init__(self, retriever: Retriever, generate: Model):
        self.retriever = GRetriever(retriever)
        self.generate = generate
        
    def __call__(self, query: str, gold_ids: list[int]) -> dict:
        result = self.retriever.search(query, gold_ids)
        answer = self.generate(
            query=query,
            documents=result.context,
        )
        return {
            "answer": answer,
            "retrieved_ids": result.ids,
        }
        