from pubinfo.pipelines.generation import build_generator
from pandas import DataFrame
from pubinfo.retrieval import Retriever
from pubinfo.retrieval.factory import build_retriever_from_config
from .config import QAConfig
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
        **config.generation_args(),
    )
    if config.prediction_mode == 'text':
        return lambda **kwargs: postprocess_text(gen(**kwargs))
    return gen

def build_qa_rag(df: DataFrame, config: QAConfig):        
    retriever = Retriever(
        df,
        config.k,
        config.columns,
        retrieve_ids=build_retriever_from_config(df, config.retrieval_config()),
    )
    generator = build_qa_generator(config)
    return RAG(retriever, generator)
