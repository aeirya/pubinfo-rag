# pubinfo/pipelines/choice_scoring.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pubinfo import template as templates

CHOICES = list("ABCD")

def softmax(scores):
    values = torch.tensor(list(scores.values()))
    probs = torch.softmax(values, dim=0).tolist()
    return dict(zip(scores.keys(), probs))

def choice_token_ids(tokenizer):
    ids = {}

    for choice in CHOICES:
        variants = [choice, " " + choice]
        ids[choice] = []

        for variant in variants:
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(token_ids) == 1:
                ids[choice].append(token_ids[0])

        if not ids[choice]:
            raise ValueError(f"No single-token variant found for {choice}")

    return ids


def get_model_name(ollama_model_name: str) -> str:
    """Return the equivalent Hugging Face repo name for an Ollama model name."""
    models = {
        "gemma2:2b": "google/gemma-2-2b-it",
        "gemma2:9b": "google/gemma-2-9b-it",
        "gemma2:27b": "google/gemma-2-27b-it",

        "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct",
        "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",

        "llama3.1:8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama3.1:70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    }

    return models.get(ollama_model_name, ollama_model_name)


def build_choice_scorer(
    prompt_template=None,
    model="google/gemma-2-2b-it",
    return_scores=False,
    device="auto",
    dtype="auto",
):
    model = get_model_name(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    llm = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype="auto",
    )

    llm.to(device)
    llm.eval()

    token_ids = choice_token_ids(tokenizer)

    @torch.inference_mode()
    def predict(**kwargs):
        text = prompt_template.format(
            query=kwargs["query"],
            documents=kwargs.get("documents", ""),
        ).rstrip()

        inputs = tokenizer(text, return_tensors="pt").to(llm.device)
        logits = llm(**inputs).logits[0, -1]

        scores = {
            choice: max(logits[i].item() for i in ids)
            for choice, ids in token_ids.items()
        }

        answer = max(scores, key=scores.get)

        if return_scores:
            return {
                "answer": answer,
                "logits": scores,
                "probs": softmax(scores),
            }

        return answer

    return predict