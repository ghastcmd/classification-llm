import argparse

import pandas as pd

from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.documents import Document

CHROMA_PATH = './chroma'

def get_embedding_function():
    return OllamaEmbeddings(model='mxbai-embed-large')

def get_dataset_quotes():
    return pd.read_csv('./dataset/dataset.csv')[['quote', 'group', 'type']]

def add_to_chroma(db, quotes):
    docs = [Document(
        page_content=quote['quote'],
        metadata={'group_type': f'{quote["group"]}/{quote["type"]}'},
        id=f'{i}',
    ) for i, quote in quotes.iterrows()]
    
    db.add_documents(docs)
    
    return db

def main(args):
    df = get_dataset_quotes()
    
    max_str_len = df.quote.str.len().max()
    
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )
    
    print(df[:10])
    
    print(df['quote'])
    
    if args.add:
        db = add_to_chroma(db, df[:10])
   
    query = 'Documentation is faulty'
    
    results = db.similarity_search(
        query,
        k=5
    )
    
    print(query)
    
    for result in results:
        print(result, result.id)

    prompt_template = """
    You have this context in the format of 'quote | group/type'.
    I want you to classify the quote within the group/type using the Context as a basis.
    
    Example:
    
    Context of the example:
    
    1. The game had a problem with delivering to the schedule. | production/schedule
    
    Description of the example:
    
    I have a problem shipping releases within schedule.
    
    Answer of the example:
    
    production/schedule
    
    Do not give answers beyond the outline of the classification as given in the exmaple: 'production/schedule'.
    Do not invent new classifications, so when prompted to say a 'group/type' use the context given
    as a source of what to say; like, if the context only says 'production/schedule' as its alternatives,
    then you must adhere to this and classify as 'production/schedule'.
    From now on it's not the example, but the actual data you need to use to answer your prompt.
    
    Context:
    
    {context}
    
    Description:
    
    {description}
    
    Answer:
    
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    model = OllamaLLM(model='phi3', temperature=0)

    chain = prompt | model
    
    context = '\n'.join([
        f'{i+1}. {result.page_content} | {result.metadata["group_type"]}'
        for i, result in enumerate(results)
    ])
    
    print('\n\ncontext\n\n', context, '\n\n====')

    ollama_result = chain.invoke({
        'description': query,
        'context': context,
    })
    
    print('\n\n====\n\nquery: ', query, '\n', ollama_result)

    previous = ollama_result
    is_different = False
    
    for _ in range(100):
        ollama_result = chain.invoke({
            'description': query,
            'context': context,
        })
        
        print(ollama_result)
        
        if ollama_result != previous:
            is_different = True
        
        previous = ollama_result
    
    if is_different:
        print('====\nfound different\n====')
    else:
        print('====\nall equal\n====')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--add', action='store_true')
    args = parser.parse_args()
    
    main(args)