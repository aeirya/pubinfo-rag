from pubinfo.pipelines.generation import build_generator
from pandas import DataFrame
from pubinfo.pipelines.qarag import QAConfig
from pubinfo.retrieval import Retriever
from .rag import RAG


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

def build_qa_rag(df: DataFrame, config: QAConfig):        
    retriever = Retriever(df, config.k, config.columns)
    generator = build_qa_generator(config)
    return RAG(retriever, generator)