from pubinfo.retrieval import Retriever
from pubinfo.pipelines.generation import build_generator
from dataclasses import dataclass, field
from pubinfo import template
from pandas import DataFrame
from pubinfo.typing import Model
from pubinfo.models import ollama
from pubinfo.dataset.publication import default_columns_no_abstract
from typing import Any

class RAGQA:
    def __init__(self, retriever: Retriever, generate: Model):
        self.retriever = retriever
        self.generate = generate

    def __call__(self, question: str) -> dict:
        result = self.retriever.search(question)

        raw = self.generate(
            query=question,
            documents=result.context,
        )

        # todo: add post processing step to raw if needed
        return {
            "answer": raw.strip(),
            "retrieved_ids": result.ids,
        }
 
@dataclass
class QAConfig:
    k: int = 5
    prompt: str = 'qa1'
    
    columns: str = 'default'
    model_args: dict[str, Any] = field(default_factory=dict)
    backend: str = 'server'
    verbose: bool = False
    
    column_list: list[str] = field(default_factory=list)

     
def build_rag_qa(df: DataFrame, config: QAConfig):
    if config.columns == 'default':
        config.column_list = None
    if config.columns == 'no_abstract':
        config.column_list = default_columns_no_abstract
        
    retriever = Retriever(df, config.k, config.column_list)
    
    model = ollama.init(**config.model_args)
    generator = build_generator(
        template=template.load(config.prompt),
        model=model,
        verbose=config.verbose,
    )
    return RAGQA(retriever, generator)