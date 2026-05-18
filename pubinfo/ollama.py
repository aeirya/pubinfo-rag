from langchain_ollama.llms import OllamaLLM

def default_args():
   # deterministic model
    return {
        "model": "gemma2:2b",
        "temperature": 0,
        "num_predict": 30,
        "top_k": 10,
        "top_p": 0.5,
        "keep_alive": "10m",
    }
    
def init_model(**kwargs):
    args = default_args()
    args.update(kwargs)
    return OllamaLLM(**args)
