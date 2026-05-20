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


#### author

from pandas import DataFrame

def clean_authors(authors: str) -> list[str]:
    authors = authors.replace("(Corresponding Author)", "")
    authors = authors.replace("(Author)", "")
    authors = authors.strip()
    author_list = re.split(r"[;,]", authors)
    # to remove weird spaces
    author_list = [' '.join(a.split()) for a in author_list]
    
    return [a.strip() for a in author_list]

def get_authors(data: DataFrame, row: int) -> list[str]:
    authors = data["authors"][row]
    return clean_authors(authors)

