import argparse
import time
import os
import re

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
unique_values = []

# DESCRIPTION_COLUMN = 'quote'
DESCRIPTION_COLUMN = 'cleaned_text'

def get_embedding_function():
    return OllamaEmbeddings(model='mxbai-embed-large')

def get_dataset_quotes():
    separated_df = pd.read_csv('./dataset/cleaned-dataset.csv')[[DESCRIPTION_COLUMN, 'group', 'type']]
    separated_df['group_type'] = separated_df["group"] + '/' + separated_df["type"]
    
    separated_df = separated_df[separated_df['group_type'] != 'management-feature/security']
    
    separated_df = separated_df.groupby(
        'group_type', group_keys=False
    ).sample(frac=0.07, random_state=123)
    
    # Removing smaller classes
    counts = separated_df['group_type'].value_counts()
    to_keep = counts[counts >= 2].index
    mask = separated_df['group_type'].isin(to_keep)
    separated_df = separated_df[mask]
    
    global unique_values
    unique_values = separated_df['group_type'].unique()
    
    train, test = train_test_split(
        separated_df, test_size=0.28, stratify=separated_df['group_type'],
        shuffle=True, random_state=42
    )
    
    return train, test

def add_to_chroma(db, quotes):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    
    docs = [Document(
        page_content=quote[DESCRIPTION_COLUMN],
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
of group/type using the provided context without inventing new classes. Do not create notes.

### Instructions:  
1. Input: A problem description related to video game development and the context of other game
development problems properly classified with its group/type.
2. Output: Only the group/type classification, with **no additional text or explanation**.  
3. Rules:  
- Classify the problem given in the Task section, **strictly based on the context above**,
do not use new group/type that is not in the context provided bellow.  
- Do not invent new groups or types.
- Prioritize the most specific and relevant classification from the context.

### Context:

| description | group/type |
| ----------- | ---------- |
{context}

### Task:
Classify the following problem description using only the context provided above:  

{description}

Output only the classification, **without** any explanation or notes or anything,
just the classification in the format of group/type.
"""
    else:
        prompt_template = """
## Section {index}

### Context:

| description | group/type |
| ----------- | ---------- |
{context}

### group/type choose one of based on the pair of description

{suggestions}

### Task:
Classify the following problem description using only the context provided
in this section:  

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
        temperature=0.0,
    )
    
    
    return model


def aggregate_prompts(arr):
    if args.overall_suggestion:
        caput = """
You are a classifier for video game development problems. Your task is to,
for each one of the sections, analyze the problem description, the given context,
and assign it a classification in the format of group/type using the provided
context of the section and studying all the classes given in the section
'Total classes', without inventing new classes.

### Total Classes

{total_classes}
"""
    else:
        caput = """
You are a classifier for video game development problems. Your task is to,
for each one of the sections, analyze the problem
description, the given context, and assign it a classification in the
format of group/type using the provided context of the section without inventing
new classes.
"""
    
    template = caput + """

### Instructions for each one of the sections:  
1. Input: A problem description related to video game development and the
context of other game development problems properly classified with
its group/type.
2. Output: Only the group/type classification, with **no additional text
or explanation**.  
3. Rules:  
- Classify the problem given in the Task section, **strictly based on the
context of the section**, do not use new group/type that is not in the context
provided in the section.  
- Do not invent new groups or types.
- Prioritize the most specific and relevant classification from the context of
the section.

---
{all_questions}
---

### Task for every one of the sections

For each one of the sections above, return a oneliner
corresponding to the group/type classification thought by you.

Output only the classification, **without** any explanation or notes or anything,
just the classification in the format of group/type."""
        

    questions = '\n---\n'.join(arr)
    
    prompt = PromptTemplate.from_template(template)
    if args.overall_suggestion:
        prompt = prompt.invoke({
            'all_questions': questions,
            'total_classes': ', '.join(unique_values),
        })
    else:
        prompt = prompt.invoke({
            'all_questions': questions,
        })
        
    
    return prompt

def main(args):
    df_train, df_test = get_dataset_quotes()
    
    db = Chroma(
        collection_name=DESCRIPTION_COLUMN,
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

    for i, entry in enumerate(test_data):
        _, entry = entry
        query = entry[DESCRIPTION_COLUMN]
        
        similarity_results = db.similarity_search(
            query,
            k=8
        )

        context, suggestions, prompt_template = prepare_prompt(similarity_results)
        
        # print(f'\n\ncontext\n\n{context}\n\n====')

        prompt_data = {
            'description': query,
            'context': context,
            'suggestions': suggestions,
            'index': i,
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
        
        def extract_formatted_string(in_str):
            matches = re.findall(r'\b[\w-]+/[\w-]+\b', in_str)
            return matches
        
        def not_trash(in_str):
            not_line = in_str != '---'
            
            not_empty =  in_str != '' 
            not_code = in_str != '```'
            not_section = False
            if in_str != '':
                not_section = in_str[0] != '#'
            
            match = extract_formatted_string(in_str)
            not_contain_class = len(match) != 0
            
            result = not_line and not_empty and not_code and not_section
            return result and not_contain_class
        
        for i, res in enumerate([extract_formatted_string(res)[0] for res in model_result_arr if not_trash(res)]):
            results[i]['result'] = res
    
    os.remove('./results.txt')

    """===== PROMPT =====
{result['prompt_txt']}"""

    for i, result in enumerate(results):
        print(f"""
            
            
==================

query: {result['query']}
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
            
        with open('results.txt', 'a+') as fp:
            res = result['result']
            true_class = result['group_type']
            suggestions = result['suggestions']
            suggestions_split = ''.join(suggestions.split(',')).split(' ')
            res_in_sug = res in suggestions_split
            true_in_sug = true_class in suggestions_split
            data_to_write = f"{res} | {true_class} | res {res_in_sug} | true {true_in_sug}\n\n{suggestions}\n\n"
            fp.write(data_to_write)
    
    
    for entry in not_matching:
        print('====== not matching ======')
        print(f'context len: {entry["context_len"]}')
        print(f'"{entry["result"][:50]}", "{entry["group_type"]}"')
        print(f'prompt len: {entry["prompt_len"]}')
        print(entry["suggestions"])

    accuracy = (len(df_test) - len(not_matching)) / len(df_test)
    result_printout = f"""
          
          Total number of tests: {len(df_test)}
          Total number of errors: {len(not_matching)}
          Accuracy: {accuracy}
    """
    
    print(result_printout)
    with open('result_printout.txt', 'w+') as fp:
        fp.write(result_printout)

    # testing_consistency(chain, query, context, ollama_result)

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--add', action='store_true')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--cached', action='store_true')
    parser.add_argument('--cleaned', action='store_true')
    parser.add_argument('--overall_suggestion', action='store_true')
    parser.add_argument('--simultaneous', action='store_true')
    
    global args
    args = parser.parse_args()
    
    global DESCRIPTION_COLUMN
    if args.cleaned:
        DESCRIPTION_COLUMN = 'cleaned_text'
    else:
        DESCRIPTION_COLUMN = 'quote'

    with open('version.txt', 'w+') as fp:
        version_values = []
        if args.add:
            version_values.append('add')
        if args.small:
            version_values.append('small')
        if args.cached:
            version_values.append('cached')
        if args.cleaned:
            version_values.append('cleaned')
        else:
            version_values.append('not cleaned')
        if args.overall_suggestion:
            version_values.append('overall_suggestion')
        if args.simultaneous:
            version_values.append('simultaneous')

        fp.write(' | '.join(version_values))

if __name__ == '__main__':
    arg_parser()    
    
    main(args)