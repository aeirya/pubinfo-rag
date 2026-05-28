# pubinfo-rag

## How to use
On one terminal session run ollama from the project root:
```
./ollama/run.sh
```

On another session you can run LLMs like:
```
python run.py \
    --query "article about career coaching" \
    --template "rag"
```

For more information, run ```python run.py -h```

## Experiments

Evaluate retrieval settings without running an LLM:
```
python scripts/eval_retrieval.py \
    --questions data/questions/abstract_questions.csv \
    --retriever tfidf \
    --columns default \
    --k 3
```

Evaluate QA RAG settings:
```
python scripts/eval_qa.py \
    --questions data/questions/abstract_questions.csv \
    --prompt qa1 \
    --retriever hybrid \
    --columns default \
    --k 4
```
