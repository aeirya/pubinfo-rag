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
        
        raw = raw.strip()
        answer = ''
        for x in 'ABCD':
            if x in raw:
                answer = x
                break
        if answer == '':
            print(f"raw text |{raw}| had no answer")
        
        # todo: add better post processing step to raw if needed
        return {
            "answer": answer,
            "retrieved_ids": result.ids,
        }
 
@dataclass
class QAConfig:
    k: int = 5
    prompt: str = 'qa1'
    
    columns: str = 'default'
    verbose: bool = False
    
    column_list: list[str] = field(default_factory=list)
    prediction: str = "choice"
    
    model: str = None
    model_args: dict[str, Any] = field(default_factory=dict)
    backend: str = 'server'
    
    def get_model_args(self):
        return {
            'template': template.load(self.prompt),
            'model': self.model,
            'backend': self.backend,
            **self.model_args
        }

def build_rag_qa(df: DataFrame, config: QAConfig):
    if config.columns == 'default':
        config.column_list = None
    if config.columns == 'no_abstract':
        config.column_list = default_columns_no_abstract
        
    retriever = Retriever(df, config.k, config.column_list)
    generator = build_generator(
        verbose=config.verbose,
        prediction_mode=config.prediction,
        **config.get_model_args()
    )
    return RAGQA(retriever, generator)