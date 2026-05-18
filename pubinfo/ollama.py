from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM

def default_args():
    '''
    maybe implement this?
    and maybe even start relying on ArgumentParser
    '''
    pass
    
def predictor(chain):
    def predict(**kwargs):
        inputs = {
            'query': kwargs.pop('query'),
            'documents': kwargs.pop('documents'),
        }
        return chain.invoke(inputs).strip()
    return predict


def init_llm(model='gemma2:2b', template=None):
    ''' 
    use ollama for invoking a local llm
    '''

    if template is None:
        template = "{query}\n\n{documents}"

    prompt = PromptTemplate.from_template(template)
   
   # deterministic model
    model = OllamaLLM(
        model=model,
        temperature=0,
        num_predict=10,
        top_k=10,
        top_p=0.5, 
        keep_alive="10m"
        )
    chain = prompt | model

    return predictor(chain)