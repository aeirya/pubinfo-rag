from langchain_core.prompts import PromptTemplate
import pubinfo as pb

def predictor(chain):
    def predict(**kwargs):
        inputs = {
            'query': kwargs.pop('query'),
            'documents': kwargs.pop('documents'),
        }
        return chain.invoke(inputs).strip()
    return predict

def init(template=None, model=None):
    ''' 
        Example:
        predict = init_llm(template='rerank', model="gemma2:2b", top_k=20, top_p=0.8)
        answer = predict(query="...", documents="...")
    '''

    if template is None:
        template = pb.prompt.default()

    if model is None:
        model = pb.ollama.init_model()

    prompt = PromptTemplate.from_template(template)
    chain = prompt | model
    return predictor(chain)