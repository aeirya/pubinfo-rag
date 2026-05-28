from langchain_core.prompts import PromptTemplate
from pubinfo import prompts
from pubinfo.models import ollama
from pubinfo.util import formatted_print


def predictor(chain, verbose=False):
    def predict(**kwargs):
        inputs = {
            'query': kwargs.pop('query'),
            'documents': kwargs.pop('documents', ''),
        }
        
        if verbose:
            print("NEXT QUESTION")
            print("PRINTING INPUT ARGS:")
            print()
            formatted_print(inputs)
            print()
        
        out = chain.invoke(inputs).strip()
        
        if verbose:
            print("OUTPUT:", out)
        
        return out
        
    return predict

def build_text_generator(template=None, model=None, verbose=False, model_args=None):
    ''' 
        Example:
        generate = build_generator(template='rerank', model="gemma2:2b", top_k=20, top_p=0.8)
        answer = generate(query="...", documents="...")
    '''

    model_args = model_args or {}

    if template is None:
        template = prompts.default()

    if verbose:
        print("template:")
        print(template)
        print()

    if isinstance(model, str):
        model = ollama.init(model=model, **model_args) 
        
    if model is None:
        model = ollama.init(**model_args)
        
    prompt = PromptTemplate.from_template(template)
    chain = prompt | model
    return predictor(chain, verbose=verbose)
