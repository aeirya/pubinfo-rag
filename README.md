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