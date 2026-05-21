from langchain_ollama.llms import OllamaLLM

local_url = '127.0.0.1:11434'
server_url = '127.0.0.1:23114'

def default_args():
    return {
        "model": "gemma2:2b",
        # deterministic model
        "temperature": 0,
        "num_predict": 30,
        "top_k": 10,
        "top_p": 0.5,
        "keep_alive": "10m",
        "reasoning": False,
        "base_url": local_url,
    }
    
def init_model(**kwargs):
    args = default_args()
    args.update({k:v for k,v in kwargs.items() if k in args and v is not None})
    return OllamaLLM(**args)

def server(
    base_url=server_url,
    **kwargs
):
    return init_model(base_url=base_url, **kwargs)

def local(
    base_url=local_url,
    **kwargs
):
    return init_model(base_url=base_url, **kwargs)