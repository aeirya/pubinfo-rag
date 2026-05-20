from typing import Callable
from pandas import DataFrame

LLM = Callable[[str], str]
Retriever = Callable[[str], list[int]]
DFRetriever = Callable[[str], DataFrame]
Question = dict[str,str]