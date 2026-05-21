from pathlib import Path

'''
module for loading prompt templates
'''

def load(templ_name, templ_dir='./templates'):
    matches = list(Path(templ_dir).glob(templ_name))
    if not matches:
        matches = list(Path(templ_dir).glob(f'*{templ_name}*'))
    if not matches:
        raise FileNotFoundError(f"No template matched {templ_name!r} in {templ_dir}")
    if len(matches) > 1:
        names = ", ".join(str(p) for p in matches)
        print(f"(Warning) Template name is ambiguous: {names}")

    file = matches[0]
    return file.read_text(encoding='utf-8')

def default():
    return "{query}\n\n{documents}"

def rerank():
    return load('rerank*')
