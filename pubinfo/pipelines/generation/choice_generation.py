from typing import Literal
import outlines
from pubinfo import prompts


Choice = Literal["A", "B", "C", "D"]
CHOICES = list('ABCD')


def build_choice_generator(
    prompt_template: str = None,
    model: str = "gemma2:2b",
    backend: str = "ollama",
):
    if prompt_template is None:
        prompt_template = prompts.default()

    if backend != "ollama":
        raise ValueError(f"Unsupported backend: {backend}")

    llm = outlines.models.ollama(model)
    choose = outlines.generate.choice(llm, CHOICES)

    def predict(**kwargs) -> Choice:
        full_prompt = prompt_template.format(
            query=kwargs["query"],
            documents=kwargs.get("documents", ""),
        )

        return choose(full_prompt).strip()

    return predict
