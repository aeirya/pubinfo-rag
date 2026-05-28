
from pubinfo.typing import GenerationMode


def build_generator(
    template: str | None = None,
    model: str | None = None,
    verbose=False,
    prediction_mode: GenerationMode = 'text',
    **kwargs
):
    model = model or "gemma2:2b"

    # choose with max likelihood
    if prediction_mode == 'choice':
        from pubinfo.pipelines.generation.choice_scoring import build_choice_scorer

        return build_choice_scorer(template, model)
    # produce tokens
    if prediction_mode == 'text':
        from pubinfo.pipelines.generation.text_generation import build_text_generator

        return build_text_generator(template, model, model_args=kwargs, verbose=verbose)
    # use external lib
    if prediction_mode == 'constrained':
        from pubinfo.pipelines.generation.choice_generation import build_choice_generator

        return build_choice_generator(template, model, backend=kwargs.get("backend", "ollama"))

    raise ValueError(f"Unknown prediction mode: {prediction_mode!r}")
