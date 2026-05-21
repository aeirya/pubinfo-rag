from pubinfo.retrieval import Retriever
from pubinfo.pipelines.generation import build_generator
from dataclasses import dataclass
from pubinfo import template
from pandas import DataFrame
from pubinfo.typing import Model
from pubinfo.models import ollama

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
    model: str = 'gemma2:2b'
    backend: str = 'server'
    verbose: bool = False

     
def build_rag_qa(df: DataFrame, config: QAConfig):
    if config.columns == 'default':
        config.columns = None
        
    retriever = Retriever(df, config.k, config.columns)
    
    model = ollama.server(model=config.model) if config.backend == 'server' else ollama.local(model=config.model)
    generator = build_generator(
        template=template.load(config.prompt),
        model=model,
        verbose=config.verbose,
    )
    return RAGQA(retriever, generator)