from pubinfo.retrieval import Retriever
from pubinfo.pipelines.generation import build_generator
from pandas import DataFrame
from pubinfo.typing import Model
from pubinfo.pipelines.qa import QAConfig
from pubinfo.retrieval import SearchResult

class RagQA:
    def __init__(self, retriever: Retriever, generate: Model):
        self.retriever = retriever
        self.generate = generate

    def __search__(self, query: str) -> SearchResult:
        return self.retriever.search(query)

    def __answer__(self, query: str, result: SearchResult):
        return self.generate(
            query=query,
            documents=result.context,
        )
        
    def __call__(self, query: str) -> dict:
        result = self.__search__(query)
        answer = self.__answer__(query, result)
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
    return RagQA(retriever, generator)