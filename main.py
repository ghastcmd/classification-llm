import argparse
import os

import pandas as pd

from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sklearn.model_selection import train_test_split
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHROMA_PATH = './chroma'

def get_embedding_function():
    return OllamaEmbeddings(model='mxbai-embed-large')

def get_dataset_quotes():
    separated_df = pd.read_csv('./dataset/dataset.csv')[['quote', 'group', 'type']]
    
    separated_df = separated_df.sample(frac=0.15, random_state=123)
    
    separated_df['group_type'] = separated_df["group"] + '/' + separated_df["type"]
    
    train, test = train_test_split(separated_df, test_size=0.2, shuffle=True, random_state=42)
    
    return train, test

def add_to_chroma(db, quotes):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        length_function=len,
        is_separator_regex=False,
    )
    
    
    docs = [Document(
        page_content=quote['quote'],
        metadata={
            'group_type': quote['group_type'],
        },
        id=f'{i}',
    ) for i, quote in quotes.iterrows()]
    
    splitted_documents = text_splitter.split_documents(docs)
    
    db.add_documents(splitted_documents)
    
    return db

def prepare_prompt(results) -> tuple[str, str, PromptTemplate]:
    prompt_template = """
You are a classifier for video game development problems. Your task is to analyze a problem
description the given context, and assign it a classification in the format
of `group/type` using the provided context without inventing new classes. Do not create notes.

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

#### group/type choose one of based on the pair of description
{suggestions}

### Task:
Classify the following problem description using only the context provided above:  
{description}

Output only the classification in the format `group/type`. 
"""
    
    prompt = PromptTemplate.from_template(prompt_template)
    
    context = '\n'.join([
        f'| {result.page_content} | {result.metadata["group_type"]} |'
        for result in results
    ])
    
    suggestions = ', '.join([result.metadata['group_type'] for result in results])
    
    return context, suggestions, prompt

def get_model():
    model = OllamaLLM(
        model='phi3',
        temperature=0,
        top_k=20,
        top_p=0.6
    )
    # model = OllamaLLM(model='phi3:14b', temperature=0)
    
    return model


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
    len_test_data = len(df_test)
    errors = 0
    
    model = get_model()

    for i, entry in enumerate(test_data):
        _, entry = entry
        query = entry['quote']
        
        results = db.similarity_search(
            query,
            k=10
        )

        context, suggestions, prompt_template = prepare_prompt(results)
        
        # print(f'\n\ncontext\n\n{context}\n\n====')

        prompt_data = {
            'description': query,
            'context': context,
            'suggestions': suggestions,
        }

        prompt = prompt_template.invoke(prompt_data)
        ollama_result = model.invoke(prompt)
        
        prompt_txt = prompt_template.format(**prompt_data)
        prompt_len = len(prompt_txt)
        group_type = entry["group_type"]
        
        print(f"""
              
              
==================

query: {query}
===== CONTEXT =====
{context}
===== PROMPT =====
{prompt_txt}
===== RESULT =====
{ollama_result}
===== TRUTH =====
{group_type}

prompt len: {prompt_len}
{i+1}/{len_test_data}""")

        
        if ollama_result.strip() != group_type:
            errors += 1
            print(f"({errors}) acc: {((i + 1) - errors) / (i + 1)}\nNOT MATCH")
            
            not_matching.append({
                'result': ollama_result,
                'group_type': group_type,
                'suggestions': suggestions,
                'context': context,
                'prompt_len': prompt_len,
            })
        else:
            print(f"({errors}) acc: {((i + 1) - errors) / (i + 1)}\nMATCH")
    
    for entry in not_matching:
        print('====== not matching ======')
        print(f'context len: {len(entry["context"])}')
        print(f"""========================
{entry['context']}
========================""")
        print(f'"{entry["result"][:50]}", "{entry["group_type"]}"')
        print(f'prompt len: {prompt_len}')
        print(entry["suggestions"])

    accuracy = (len(df_test) - len(not_matching)) / len(df_test)
    print(f"""
          
          Total number of tests: {len(df_test)}
          Total number of errors: {len(not_matching)}
          Accuracy: {accuracy}
    """)

    # testing_consistency(chain, query, context, ollama_result)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--add', action='store_true')
    args = parser.parse_args()
    
    main(args)