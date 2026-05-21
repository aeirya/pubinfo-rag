from langchain_core.prompts import PromptTemplate
import pubinfo as pb
from pubinfo.util import format_question, formatted_print


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
        
        if verbose or True:
            print("OUTPUT:", out)
        
        return out
        
    return predict

def build_text_generator(template=None, model=None, verbose=False, model_args={}):
    ''' 
        Example:
        generate = build_generator(template='rerank', model="gemma2:2b", top_k=20, top_p=0.8)
        answer = generate(query="...", documents="...")
    '''

    if template is None:
        template = pb.prompt.default()

    if verbose:
        print("template:")
        print(template)
        print()

    if isinstance(model, str):
        model = ollama.init(model=model, **model_args) 
        
    if model is None:
        model = pb.ollama.init_model()
        
    prompt = PromptTemplate.from_template(template)
    chain = prompt | model
    return predictor(chain, verbose=verbose)