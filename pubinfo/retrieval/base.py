from dataclasses import dataclass
import pandas as pd
from pubinfo.template.format import rows_to_context
from pubinfo.retrieval import build_retriever


@dataclass
class SearchResult:
    ids: list
    context: str


class Retriever:
    def __init__(self, df: pd.DataFrame, k: int = 4, columns: list|str = None):
        self.df = df
        self.k = k
        self.columns = columns
        self._retrieve_ids = build_retriever(df, k=k, columns=columns)

    def ids(self, query: str) -> list:
        return self._retrieve_ids(query)

    def to_context(self, ids):
        return rows_to_context(self.df, ids, columns=self.columns)

    def search(self, query: str) -> SearchResult:
        ids = self.ids(query)
        context = self.to_context(ids)
        return SearchResult(ids=ids, context=context)