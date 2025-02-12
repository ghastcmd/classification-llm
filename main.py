import argparse
import time
import os

import pandas as pd

from langchain_core.prompts import PromptTemplate
from langchain_core.prompt_values import PromptValue
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.model_selection import train_test_split

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import dotenv_values

CHROMA_PATH = './chroma'
args = {}

def get_embedding_function():
    return OllamaEmbeddings(model='mxbai-embed-large')

def get_dataset_quotes():
    separated_df = pd.read_csv('./dataset/dataset.csv')[['quote', 'group', 'type']]
    
    separated_df = separated_df.sample(frac=0.07, random_state=123)
    
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
    if args.small:
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

Output only the classification, **without** any explanation or notes or anything,
just the classification in the format of `group/type`.
"""
    else:
        prompt_template = """
### Context:

| description | group/type |
| ----------- | ---------- |
{context}

#### group/type choose one of based on the pair of description

{suggestions}

### Task:
Classify the following problem description using only the context provided above:  

{description}

"""
        
    
    prompt = PromptTemplate.from_template(prompt_template)
    
    context = '\n'.join([
        f'| {result.page_content} | {result.metadata["group_type"]} |'
        for result in results
    ])
    
    suggestions = ', '.join([result.metadata['group_type'] for result in results])
    
    return context, suggestions, prompt

def get_model():
    # model = OllamaLLM(
    #     model='phi3.5',
    #     temperature=0,
    #     top_k=20,
    #     top_p=0.4
    # )
    # model = OllamaLLM(model='phi3:14b', temperature=0)
    if 'GOOGLE_API_KEY' not in os.environ:
        os.environ['GOOGLE_API_KEY'] = dotenv_values('.env')['GEMNINI_API']
    
    model = ChatGoogleGenerativeAI(
        model='gemini-2.0-pro-exp',
    )
    
    
    return model


def aggregate_prompts(arr):
    template = """
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

---
{all_questions}
---

### Task for every section delimited by '---'

For each of the given sections delimited by a line '---', return a
oneliner corresponding to the group/type classification reasoned by you.

There is a total of {total_prompts} of sections, so there should be a total
of {total_prompts} oneliner answers.

Output only the classification, **without** any explanation or notes or anything,
just the classification in the format of `group/type`.
"""

    questions = '\n---\n'.join(arr)
    
    prompt = PromptTemplate.from_template(template)
    prompt = prompt.invoke({
        'all_questions': questions,
        'total_prompts': len(arr),
    })
    
    return prompt

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

    results = []

    for entry in test_data:
        _, entry = entry
        query = entry['quote']
        
        similarity_results = db.similarity_search(
            query,
            k=10
        )

        context, suggestions, prompt_template = prepare_prompt(similarity_results)
        
        # print(f'\n\ncontext\n\n{context}\n\n====')

        prompt_data = {
            'description': query,
            'context': context,
            'suggestions': suggestions,
        }

        prompt = prompt_template.invoke(prompt_data)
        
        prompt_txt = prompt_template.format(**prompt_data)
        prompt_len = len(prompt_txt)
        group_type = entry["group_type"]
        
        result = {
            'query': query,
            'prompt_txt': prompt_txt,
            'group_type': group_type,
            'prompt_len': prompt_len,
            'suggestions': suggestions,
            'context': context,
        }
        
        if args.small:
            print('invoked small part')
            model_result = model.invoke(prompt)
            model_result = model_result.content
            result['result'] = model_result
        else:
            result['prompt'] = prompt.to_string()
    
        results.append(result)
    
    if not args.small:
        prompt = aggregate_prompts([res['prompt'] for res in results])
        
        with open('prompt.txt', 'w+') as fp:
            fp.write(prompt.to_string())
        
        if not args.cached:
            model_result = model.invoke(prompt)
            model_str = model_result.content
            with open('cached.txt', 'w+') as fp:
                fp.write(model_str)
        else:
            with open('cached.txt', 'r') as fp:
                model_str = fp.read()
        model_result_arr = model_str.strip().split('\n')
        print(model_result_arr)
        for i, res in enumerate([res for res in model_result_arr if res != '---']):
            results[i]['result'] = res
    
    for i, result in enumerate(results):
        print(f"""
            
            
==================

query: {result['query']}
===== PROMPT =====
{result['prompt_txt']}
===== RESULT =====
{result['result']}
===== TRUTH =====
{result['group_type']}

prompt len: {result['prompt_len']}
{i+1}/{len_test_data}""")

        
        if result['result'].strip() != result['group_type']:
            errors += 1
            print(f"({errors}) acc: {((i + 1) - errors) / (i + 1)}\nNOT MATCH")
            
            not_matching.append({
                'result': result['result'],
                'group_type': result['group_type'],
                'suggestions': result['suggestions'],
                'prompt_len': result['prompt_len'],
                'context_len': len(result['context']),
            })
        else:
            print(f"({errors}) acc: {((i + 1) - errors) / (i + 1)}\nMATCH")
            
    
    
    for entry in not_matching:
        print('====== not matching ======')
        print(f'context len: {entry["context_len"]}')
        print(f'"{entry["result"][:50]}", "{entry["group_type"]}"')
        print(f'prompt len: {entry["prompt_len"]}')
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
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--cached', action='store_true')
    args = parser.parse_args()
    
    main(args)