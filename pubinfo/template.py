from pathlib import Path

def load(templ_name, templ_dir='./templates'):
    file = next(iter(Path(templ_dir).glob(templ_name)))
    return file.read_text(encoding='utf-8')

def rerank():
    return load('rerank*')
