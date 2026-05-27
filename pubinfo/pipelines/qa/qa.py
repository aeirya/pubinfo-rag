from pubinfo.retrieval import Retriever
from pubinfo.pipelines.generation import build_generator
from pandas import DataFrame
from pubinfo.typing import Model
from pubinfo.pipelines.qa import QAConfig


class RAGQA:
    def __init__(self, retriever: Retriever, generate: Model):
        self.retriever = retriever
        self.generate = generate

    def __call__(self, question: str) -> dict:
        result = self.retriever.search(question)
        answer = self.generate(
            query=question,
            documents=result.context,
        )
        return {
            "answer": answer,
            "retrieved_ids": result.ids,
        }
 
def postprocess_text(raw):
    raw = raw.strip()
    for x in 'ABCD':
        if x in raw:
            return x
    return ''

def build_qa_generator(config: QAConfig): 
    gen = build_generator(
        verbose=config.verbose,
        prediction_mode=config.prediction_mode,
        **config.model_args
    )
    if config.prediction_mode == 'text':
        return lambda x: postprocess_text(gen(x))
    return gen

def build_rag_qa(df: DataFrame, config: QAConfig):        
    retriever = Retriever(df, config.k, config.columns)
    generator = build_qa_generator(config)
    return RAGQA(retriever, generator)