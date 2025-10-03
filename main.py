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
unique_groups = []
group_type_dict = {}

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
    
    global unique_values, unique_groups
    unique_values = separated_df['group_type'].unique()
    unique_groups = separated_df['group'].unique()
    
    for val in unique_values:
        group, ttype = val.split('/')
        group_type_dict.setdefault(group, []).append([ttype, f'{group}/{ttype}'])
        
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

ALL_CLASSES_DESCRIPTION = """
| Group | Type | Description of the problem |
| production | design | Any problem regarding the design of the game, like balancing the gameplay. Not a technical detail. |
| | documentation | Not planning the game beforehand, not documenting the code, artifacts or game plan. |
| | tools | Any problem with tools like engines, APIs, development kits, third-party software, etc. |
| | technical | Problems with the team code/assets infrastructure. |
| | testing | Any problem regarding the testing. |
| | bugs | When there are too many bugs in the game/engine, any failure in the game design or technical issues. |
| | prototyping | Lack of or no prototyping phase nor validation of the gameplay/feature. |
| management | unrealistic-scope | Planning too many features that end up impossible to implement it in a reasonable time. |
| | feature-creep | Adding unplanned new features to the game during its implementation. |
| | cutting-features | Cutting features previously planned because of other factor like short deadlines. |
| | delays | Problems regarding any delay in the production. |
| | crunch-time | When developers continuously spent extra hours working in the project. |
| | communication | Problems regarding communication with any stakeholder. |
| | team |  Problems in setting up the team, loss of professionals during the development or outsourcing. |
| | over-budget | Project cost more money than expected. |
| | multiple-projects | When there is more than one project being developed at the same time. |
| | planning | Problems involving too much time planing/scheduling or the lack of it. |
| | security | Problems regarding leaked assets. |
| business | marketing | Problems regarding marketing/advertising |
| | monetization | Problems with the process used to generate revenue from a video game product.|
"""

ALL_CLASSES_DESCRIPTION_2 = """
| group/type | description of the class |
| production/design | Any problem regarding the design of the game, like balancing the gameplay. Not a technical detail. |
| production/documentation | Not planning the game beforehand, not documenting the code, artifacts or game plan. |
| production/tools | Any problem with tools like engines, APIs, development kits, third-party software, etc. |
| production/technical | Problems with the team code/assets infrastructure. |
| production/testing | Any problem regarding the testing. |
| production/bugs | When there are too many bugs in the game/engine, any failure in the game design or technical issues. |
| production/prototyping | Lack of or no prototyping phase nor validation of the gameplay/feature. |
| management/unrealistic-scope | Planning too many features that end up impossible to implement it in a reasonable time. |
| management/feature-creep | Adding unplanned new features to the game during its implementation. |
| management/cutting-features | Cutting features previously planned because of other factor like short deadlines. |
| management/delays | Problems regarding any delay in the production. |
| management/crunch-time | When developers continuously spent extra hours working in the project. |
| management/communication | Problems regarding communication with any stakeholder. |
| management/team |  Problems in setting up the team, loss of professionals during the development or outsourcing. |
| management/over-budget | Project cost more money than expected. |
| management/multiple-projects | When there is more than one project being developed at the same time. |
| management/planning | Problems involving too much time planing/scheduling or the lack of it. |
| management/security | Problems regarding leaked assets. |
| business/marketing | Problems regarding marketing/advertising |
| business/monetization | Problems with the process used to generate revenue from a video game product.|
"""

def aggregate_prompts(arr):
    if args.overall_suggestion:
        if not args.segmented:
            
            if args.description:
                caput = """
You are a classifier for video game development problems.

Your task is to, for each one of the sections named with the format "## question 0"
(with two hashtag '##', followed by the name 'question' with the enumeration of the section,
in the given example it is 0, so this corresponds to the first question section), you'll
classify its description.

So, for instance, if you are analysing question 0, you'll analyse only the context of the
question 0, and classify it in the format of group/type using only the context of the 
section question 0 and the classes from the section 'classes from the dataset',
without inventing new classes or using classes from outside the scope of the section 'question 0',
or the section 'classes from the dataset'. Thus, if you want to classify the description of
question 0, do not use the context from other question sections.

If you can't figure out the classification only with the context of the section question 0,
feel free to classify as what it is most likely to be given the context of question 0 or
using the classes contained in the section 'classes from the dataset'.

The section 'classes from the dataset' contains a table of which the first column is the class
and the second column is the description of said class, thus, each class have its description
and you must analyse the description of the classes to best classify the question description.

I've given the example of question 0, but this also holds for every other question, like
question 1, question 2, and so on.

## classes from the dataset

{total_classes}
    """
            else:
                caput = """
You are a classifier for video game development problems.

Your task is to, for each one of the sections named with the format "## question 0"
(with two hashtag '##', followed by the name 'question' with the enumeration of the section,
in the given example it is 0, so this corresponds to the first question section), you'll
classify its description.

So, for instance, if you are analysing question 0, you'll analyse only the context of the
question 0, and classify it in the format of group/type using only the context of the 
section question 0 without inventing new classes or using classes from outside the scope
of the section 'question 0'. Thus, if you want to classify the description of question 0,
do not use the context from other question sections.

If you can't figure out the classification only with the context of the section question 0,
feel free to classify as what it is most likely to be given the context of question 0 or
using the classes contained in the section 'classes from the dataset'.

I've given the example of question 0, but this also holds for every other question, like
question 1, question 2, and so on.

## classes from the dataset

{total_classes}
    """
        elif not args.second: # ! first segmented (group)
                caput = """
You are a classifier for video game development problems.

Your task is to, for each one of the sections named with the format "## question 0"
(with two hashtag '##', followed by the name 'question' with the enumeration of the section,
in the given example it is 0, so this corresponds to the first question section), you'll
classify its description.

So, for instance, if you are analysing question 0, you'll analyse only the context of the
question 0, and classify it in the format of group using only the context of the 
section question 0 and with data from the section 'classes from the dataset', without
inventing new classes or using classes from outside the scope of the section 'question 0'.

Thus, if you want to classify the description of question 0, do not use the context from
other question sections.

I've given the example of question 0, but this also holds for every other question, like
question 1, question 2, and so on.

## classes from the dataset

{total_classes}
"""
        else: # ! is second of segmented (type -> group/type)
                caput = """
You are a classifier for video game development problems.

Your task is to, for each one of the sections named with the format "## question 0"
(with two hashtag '##', followed by the name 'question' with the enumeration of the section,
in the given example it is 0, so this corresponds to the first question section), you'll
classify its description.

So, for instance, if you are analysing question 0, you'll analyse only the context of the
question 0, and classify it in the format of group/type using only the context of the 
section question 0 without inventing new classes or using classes from outside the scope
of the section 'question 0'.

Thus, if you want to classify the description of question 0, do not use the context from
other question sections.

I've given the example of question 0, but this also holds for every other question, like
question 1, question 2, and so on.
"""

    else: # ! without overall suggestions
        caput = """
You are a classifier for video game development problems.

Your task is to, for each one of the sections named with the format "## question 0"
(with two hashtag '##', followed by the name 'question' with the enumeration of the section,
in the given example it is 0, so this corresponds to the first question section), you'll
classify its description.

So, for instance, if you are analysing question 0, you'll analyse only the context of the
question 0, and classify it in the format of group/type using only the context of the 
section question 0 without inventing new classes or using classes from outside the scope
of the section 'question 0'. Thus, if you want to classify the description of question 0,
do not use the context from other question sections.

If you can't figure out the classification only with the context of the section question 0,
feel free to classify as what it is most likely to be given the context of question 0.

I've given the example of question 0, but this also holds for every other question, like
question 1, question 2, and so on.
"""




    if not args.segmented:
        everything_else = """
## Instructions for each one of the sections:  
1. Input: A problem description related to video game development and the
context of other game development problems properly classified with
its group/type.
2. Output: A header with the enumeration of the section, and the group/type
classification, with **no additional text or explanation**.
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

## Task for every one of the sections

For each one of the sections above, return a oneliner
corresponding to the group/type classification thought by you.

Output only the classification, **without** any explanation or notes or anything,
just the classification in the format of group/type."""

    elif not args.second: # ! is **first** of segmented
            everything_else = """
## Instructions for each one of the sections:  
1. Input: A problem description related to video game development.
2. Output: A header with the enumeration of the section, and the group
classification, with **no additional text or explanation**.
3. Rules:
- Prioritize the most specific and relevant classification of the section.

---
{all_questions}
---

## Task for every one of the sections

For each one of the sections above, return a oneliner
corresponding to the group classification thought by you.

Output only the classification, **without** any explanation or notes or anything,
just the classification in the format of group."""

    else: # ! is second of segmented
            everything_else = """
## Instructions for each one of the sections:  
1. Input: A problem description related to video game development.
2. Output: A header with the enumeration of the section, and the group/type
classification, with **no additional text or explanation**.
3. Rules:
- Prioritize the most specific and relevant classification of the section.

---
{all_questions}
---

## Task for every one of the sections

For each one of the sections above, return a oneliner
corresponding to the group/type classification thought by you.

Output only the classification, **without** any explanation or notes or anything,
just the classification in the format of group/type."""

    template = f"""{caput}

{everything_else}"""

    questions = '\n---\n'.join(arr)
    
    prompt = PromptTemplate.from_template(template)
    if args.overall_suggestion and not args.second:
        if not args.segmented:
            if args.description:
                prompt = prompt.invoke({
                    'all_questions': questions,
                    'total_classes': ALL_CLASSES_DESCRIPTION_2,
                })
            else:
                prompt = prompt.invoke({
                    'all_questions': questions,
                    'total_classes': ', '.join(unique_values),
                })
        else:
            prompt = prompt.invoke({
                'all_questions': questions,
                'total_classes': ', '.join(unique_groups),
            })
    else:
        prompt = prompt.invoke({
            'all_questions': questions,
        })
        
    
    return prompt

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
        if not args.without_few_shot:
            prompt_template = """
## question {index}

### Context:

| description | group/type |
| ----------- | ---------- |
{context}

### Task:
Classify the following problem description using only the context provided
in this section:  

{description}

"""
        else:
            if not args.segmented:
                prompt_template = """
## question {index}

### Task:
Classify the following problem description using only the context provided
in this section:  

{description}

"""
            else:
                if not args.second:
                    prompt_template = """
## question {index}

### Task:
Classify the following problem description using only the context provided
in this section:

{description}

"""
                else:
                    prompt_template = """
## question {index}

| type | group/type |
| ---- | ---------- |
{type_2_group}

### Task:
Classify the following problem description using only the context provided
in this section:

{description}

"""
                    
        
    
    prompt = PromptTemplate.from_template(prompt_template)
    
    context = ''
    
    if not args.without_few_shot:
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

def get_results_form(model_str: str, regex_format: str):
    model_result_arr = model_str.strip().split('\n')
    print(model_result_arr)
    
    def extract_formatted_string(in_str):
        matches = re.findall(regex_format, in_str)
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
    
    return [extract_formatted_string(res)[0] for res in model_result_arr if not_trash(res)]

def get_type_table(group: str):
    global group_type_dict
    table_values = '\n'.join([f'| {entry[0]} | {entry[1]} |' for entry in group_type_dict[group]])
    return table_values

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

    if args.second:
        previous_model_out = ''
        with open('to_second.txt', 'r') as fp:
            previous_model_out = fp.read()
        
        groups = get_results_form(previous_model_out, r'\b[\w-]+\b')

    for i, entry in enumerate(test_data):
        _, entry = entry
        query = entry[DESCRIPTION_COLUMN]
        
        if args.second:
            group_entry = groups[i]
        
        similarity_results = []
        
        if not args.without_few_shot:
            similarity_results = db.similarity_search(
                query,
                k=8
            )

        context, suggestions, prompt_template = prepare_prompt(similarity_results)
        
        # print(f'\n\ncontext\n\n{context}\n\n====')

        if not args.without_few_shot:
            prompt_data = {
                'description': query,
                'context': context,
                'suggestions': suggestions,
                'index': i,
            }
        else:
            if args.segmented and args.second:
                prompt_data = {
                    'description': query,
                    'suggestions': suggestions,
                    'type_2_group': get_type_table(group_entry),
                    'index': i,
                }
            else:
                if not args.segmented:
                    prompt_data = {
                        'description': query,
                        'suggestions': suggestions,
                        'index': i,
                    }
                else:
                    prompt_data = {
                        'description': query,
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
            'group': entry['group'],
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
            if not args.second:
                with open('to_second.txt', 'w+') as fp:
                    fp.write(model_str)
        else:
            with open('cached.txt', 'r') as fp:
                model_str = fp.read()

        if not args.segmented or args.second:
            for i, res in enumerate(get_results_form(model_str, r'\b[\w-]+/[\w-]+\b')):
                results[i]['result'] = res
        else:
            for i, res in enumerate(get_results_form(model_str, r'\b[\w-]+\b')):
                results[i]['result'] = res
    
    
    if os.path.exists('./results.txt'):
        os.remove('./results.txt')

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

        def get_comparison_safe():
            if not args.segmented or args.second:
                return result['group_type']
            else:
                return result['group']
        
        if result['result'].strip() != get_comparison_safe():
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
            correct = res == true_class
            data_to_write = f"{res} | {true_class} | res {res_in_sug} | orig {true_in_sug} | c {correct}\n\n{suggestions}\n\n"
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
    parser.add_argument('--segmented', action='store_true')
    parser.add_argument('--second', action='store_true')
    parser.add_argument('--without_few_shot', action='store_true')
    parser.add_argument('--description', action='store_true')
    
    global args
    args = parser.parse_args()
    
    global DESCRIPTION_COLUMN
    if args.cleaned:
        DESCRIPTION_COLUMN = 'cleaned_text'
    else:
        DESCRIPTION_COLUMN = 'quote'

    if args.without_few_shot and not args.overall_suggestion:
        print('Incompatible argument options: |--without_few_shot| and |--overall_suggestion|')
        exit(1)

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
        if args.segmented:
            version_values.append('segmented')
        if args.second:
            version_values.append('second segmented')
        if args.without_few_shot:
            version_values.append('without_few_shot')
        if args.description:
            version_values.append('description')

        fp.write(' | '.join(version_values))

if __name__ == '__main__':
    arg_parser()    
    
    main(args)