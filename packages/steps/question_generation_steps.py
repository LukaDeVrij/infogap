import ipdb
from typing import List
import os  
import polars as pl

from .info_diff_steps import load_other_client

# {{article context Cp}}
# After reading the sentence {{anchor sentence Sk}}, ask 5
# questions about a part of this sentence that you are curious
# about which you don’t have an answer for.

# precondition: paragraph_facts has at least two facts
def generate_questions(client, paragraph_facts: List[str], 
                       person_name: str, 
                       model_name: str,
                       language: str):
    assert len(paragraph_facts) >= 2, "paragraph_facts should have at least two facts"
    context = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(paragraph_facts[0:-1])])
    anchor = paragraph_facts[-1]
    if language == 'en':
        prefix = f"Consider the following facts regarding the life of {person_name}:\n\n{context}\n\n"
        prompt = " ".join([
            prefix,
            f"After reading the sentence '{anchor}', ask 5 questions about a part of this sentence that you are curious about which you don’t have an answer for. Return the questions as a JSON list."
        ])
    elif language == 'fr':
        prefix = f"Considérez les faits suivants concernant la vie de {person_name}.\n\n{context}\n\n"
        prompt = " ".join([
            prefix,
            f"Après avoir lu la phrase '{anchor}', posez 5 questions sur une partie de cette phrase qui vous intrigue et pour laquelle vous n'avez pas de réponse. Retournez les questions sous forme de liste JSON."
        ])
    elif language == 'ru':
        prefix = f"Рассмотрите следующие факты о жизни {person_name}.\n\n{context}\n\n"
        prompt = " ".join([
            prefix,
            f"Прочитав предложение '{anchor}', задайте 5 вопросов о части этого предложения, которая вас интересует, но на которую у вас нет ответа. Верните вопросы в виде списка JSON."
            ])
    else:
        raise ValueError(f"Unsupported language: {language}")
    print(f"Prompt: {prompt}")
    message=[{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
            model=model_name,
            max_tokens=len(prompt) + 100,
            temperature=0,
        messages = message)
    response_content = response.choices[0].message.content
    return response_content

def generate_questions_discourse_level(client, paragraph_facts: List[str], 
                       person_name: str, 
                       model_name: str,
                       language: str):
    assert len(paragraph_facts) >= 2, "paragraph_facts should have at least two facts"
    context = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(paragraph_facts[0:-1])])
    anchor = paragraph_facts[-1]
    if language == 'en':
        prefix = f"Consider the following facts regarding the life of {person_name}:\n\n{context}\n\n"
        prompt = " ".join([
            prefix,
            f"Ask a question where someone might have answered with this discourse about {person_name}."
        ])
    elif language == 'fr':
        prefix = f"Considérez les faits suivants concernant la vie de {person_name}.\n\n{context}\n\n"
        prompt = " ".join([
            prefix,
            f"Après avoir lu la phrase '{anchor}', posez 5 questions sur une partie de cette phrase qui vous intrigue et pour laquelle vous n'avez pas de réponse. Retournez les questions sous forme de liste JSON."
        ])
    elif language == 'ru':
        prefix = f"Рассмотрите следующие факты о жизни {person_name}.\n\n{context}\n\n"
        prompt = " ".join([
            prefix,
            f"Прочитав предложение '{anchor}', задайте 5 вопросов о части этого предложения, которая вас интересует, но на которую у вас нет ответа. Верните вопросы в виде списка JSON."
            ])
    else:
        raise ValueError(f"Unsupported language: {language}")
    print(f"Prompt: {prompt}")
    message=[{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
            model=model_name,
            max_tokens=len(prompt) + 100,
            temperature=0,
        messages = message)
    response_content = response.choices[0].message.content
    return response_content