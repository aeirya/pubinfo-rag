
def model_args_list():
    return [
        {
            'model': 'gemma2:2b',
            'num_ctx': 2048,
            'num_predict': 5,
            'reasoning': False,
        },
        {
            'model': 'gemma2:2b',
            "num_ctx": 4096,
            'num_predict': 128,
            'reasoning': False,
        },
        {
            'model': 'qwen2.5:7b',
            "num_ctx": 4096,
            'num_predict': 256,
            'reasoning': False
        },
        {
            'model': "qwen3:8b",
            'num_ctx': 2048*2,
            'reasoning': True,
            'num_predict': 256,
        }
    ]
    
IRS = [
    "semantic", "tfidf", "hybrid"
]

RETRIEVAL_SCENARIOS = [
    "no-rag", "normal", "guided"
]

def limit_options():
    pass
   