from langchain_core.documents import Document
from pandas import DataFrame

def format_results(retrieved_docs: list[Document], data: DataFrame):
    context = "<Articles found by the search engine>\n"

    for doc in retrieved_docs:
        row = doc.metadata.get('row_id')

        context += "Title:\n"
        context += str(data["title"][row]) + "\n"
        context += "Authors:\n"
        context += str(data["authors"][row]) + "\n"
        context += "Publication date:\n"
        context += str(data["date_published"][row]) + "\n"

    context += "</Articles found by the search engine>"

    return context