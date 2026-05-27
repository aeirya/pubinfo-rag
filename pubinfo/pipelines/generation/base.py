from typing import Literal

from pubinfo.pipelines.generation.choice_generation import build_choice_generator
from pubinfo.pipelines.generation.choice_scoring import build_choice_scorer
from pubinfo.pipelines.generation.text_generation import build_text_generator

GenerationMode = Literal['constrained', 'text', 'choice']

def build_generator(
    template: str,
    model: str,
    verbose = False,
    prediction_mode: GenerationMode = 'text',
    **kwargs
):
    # choose with max likelihood
    if prediction_mode == 'choice':
        return build_choice_scorer(template, model)
    # produce tokens
    if prediction_mode == 'text':
        return build_text_generator(template, model, model_args=kwargs, verbose=verbose)
    # use external lib
    if prediction_mode == 'constrained':
        return build_choice_generator(template, model)