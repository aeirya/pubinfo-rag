from typing import Callable
from pandas import DataFrame
from typing import Protocol


LLM = Callable[[str], str]

QAModel = Callable[[str], dict[str, str]]

Retriever = Callable[[str], list[int]]
DFRetriever = Callable[[str], DataFrame]
Question = dict[str,str]

class Model(Protocol):
    def __call__(self, *, query: str, documents: str) -> str: ...