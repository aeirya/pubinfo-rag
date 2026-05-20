from pubinfo.retrieval import build_df_retriver
from pubinfo.template import context

def init(model, retriever, df):
    def invoke(query):
        hit_ids = retriever(query)
        answer = model(
            query=query, 
            documents=context.format(df.iloc[hit_ids])
        )
        return {
            'answer': answer,
            'retrieved_ids': hit_ids
        }
        
    return invoke
