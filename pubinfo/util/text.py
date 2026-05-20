import re
from html import unescape

def clean_abstract(text):
    if not text:
        return None

    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def quote(s: str):
    return f'\"{s}\"'