from pubinfo.pipelines.generation import build_generator
from pandas import DataFrame
from pubinfo.models import ollama
from pubinfo import template

def build_dummy_qa(df: DataFrame, prompt='dummy_qa', model='gemma2:2b', verbose=False, backend='server'):
    model = ollama.init(model=model)
    generator = build_generator(
        template=template.load(prompt),
        model=model,
        verbose=verbose,
    )
    return lambda query: { 
        'answer': generator(query=query),
        'retrieved_ids': [], 
        }
