import re
from html import unescape

from pubinfo.util.text import quote
from pandas import DataFrame

def clean_abstract(text):
    if not text: return None
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_authors(authors: str) -> list[str]:
    authors = authors.replace("(Corresponding Author)", "")
    authors = authors.replace("(Author)", "")
    authors = authors.strip()
    author_list = re.split(r"[;,]", authors)
    # to remove weird spaces
    author_list = [' '.join(a.split()) for a in author_list]
    return ', '.join([a.strip() for a in author_list])

def clean_date(date: str):
    # todo: do tests on the effect of date format
    return date

def clean_publication_data(df: DataFrame):
    if 'abstract' in df:
        df['abstract'] = df['abstract'].apply(clean_abstract).apply(quote)
    if 'authors' in df:
        df['authors'] = df['authors'].apply(clean_authors)
    return df

def get_clean_authors(data: DataFrame, row: int) -> list[str]:
    authors = data["authors"][row]
    return clean_authors(authors).split(', ')

