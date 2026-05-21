from langchain_ollama.llms import OllamaLLM

local_url = '127.0.0.1:11434'
server_url = '127.0.0.1:23114'

def default_args():
    return {
        "model": "gemma2:2b",
        # deterministic model
        
        "keep_alive": "10m",
        "reasoning": False,
        "base_url": local_url,
        
        "num_predict": 30,
        # "num_ctx": 4096,
        
        "temperature": 0,
        "seed": 42,
        # "top_k": 10,
        # "top_p": 0.5,
    }
    
def __init_model__(**kwargs):
    args = default_args()
    args.update({k:v for k,v in kwargs.items() if k in args and v is not None})
    return OllamaLLM(**args)

def server(
    base_url=server_url,
    **kwargs
):
    return __init_model__(base_url=base_url, **kwargs)

def local(
    base_url=local_url,
    **kwargs
):
    return __init_model__(base_url=base_url, **kwargs)

def init(backend='server', **kwargs):
    if backend == 'server':
        return server(**kwargs)
    if backend == 'local':
        return local(**kwargs)