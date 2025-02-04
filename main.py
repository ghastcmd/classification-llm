import argparse
import os

import pandas as pd

from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sklearn.model_selection import train_test_split

CHROMA_PATH = './chroma'

def get_embedding_function():
    return OllamaEmbeddings(model='mxbai-embed-large')

def get_dataset_quotes():
    separated_df = pd.read_csv('./dataset/dataset.csv')[['quote', 'group', 'type']]
    
    separated_df = separated_df.sample(frac=0.05, random_state=123)
    
    separated_df['group_type'] = separated_df["group"] + '/' + separated_df["type"]
    
    train, test = train_test_split(separated_df, test_size=0.2, shuffle=True, random_state=42)
    
    return train, test

def add_to_chroma(db, quotes):
    docs = [Document(
        page_content=quote['quote'],
        metadata={'group_type': quote['group_type']},
        id=f'{i}',
    ) for i, quote in quotes.iterrows()]
    
    db.add_documents(docs)
    
    return db

class Chain:
    x = 0

def prepare_prompt(results) -> tuple[Chain, str]:
    prompt_template = """
You are a classifier for video game development problems. Your task is to analyze a problem
description and assign it a classification in the format of `group/type` using only the provided
context.

### Instructions:  
1. Input: A problem description related to video game development and the context of other game
development problems properly classified with its `group/type`.
2. Output: Only the `group/type` classification, with **no additional text or explanation**.  
3. Rules:  
   - Classify the problem given in the Task section, **strictly based on the context above**,
   do not use new `group/type` that is not in the context provided bellow.  
   - Do not invent new groups or types.
   - Prioritize the most specific and relevant classification from the context.

### Context:
| description | group/type |
| ----------- | ---------- |
{context}

### Task:
Classify the following problem description using only the context provided above:  
{description}

Output only the classification in the format `group/type`. 
"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    os.environ['OLLAMA_NOHISTORY'] = '1'
    model = OllamaLLM(model='phi3', temperature=0)

    chain = prompt | model
    
    context = '\n'.join([
        f'| {result.page_content} | {result.metadata["group_type"]} |'
        for result in results
    ])
    
    return chain, context, [result.metadata['group_type'] for result in results]

def testing_consistency(
    chain: Chain,
    query: str,
    context: str,
    ollama_result: str
):
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

def main(args):
    df_train, df_test = get_dataset_quotes()
    
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )
    
    if args.add:
        db = add_to_chroma(db, df_train)

    not_matching = []

    # test_data = df_test.iloc[:1].iterrows()
    test_data = df_test.iterrows()

    for _, entry in test_data:
        query = entry['quote']
        
        results = db.similarity_search(
            query,
            k=5
        )
        
        chain, context, input_classes = prepare_prompt(results)
        
        # print(f'\n\ncontext\n\n{context}\n\n====')

        ollama_result = chain.invoke({
            'description': query,
            'context': context,
        })
        
        group_type = entry["group_type"]
        
        print(f'\n\n====\n\nquery: {query}\nRESULT\n{ollama_result}\nTRUTH\n{group_type}\n')
        
        if ollama_result != group_type:
            print("NOT MATCH")
            not_matching.append((ollama_result, group_type, input_classes, context))
        else:
            print("MATCH")
    
    for entry in not_matching:
        print(f'"{entry[0][:50]}", "{entry[1]}"')
        print(entry[2])
        # print(f'=======\n{entry[3]}\n=======')

    print(f'\nTotal number of tests: {len(df_test)}\nTotal number of errors: {len(not_matching)}')

    # testing_consistency(chain, query, context, ollama_result)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--add', action='store_true')
    args = parser.parse_args()
    
    main(args)