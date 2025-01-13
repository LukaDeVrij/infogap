import openai
from dotenv import dotenv_values
import loguru
import ipdb
from typing import List, Tuple
from dotenv import dotenv_values
# import dataclasses    
from dataclasses import dataclass

config = dotenv_values(".env")
key = config["THE_KEY"] 
logger = loguru.logger

@dataclass
class FactParagraph:
    """
    A list of facts that are contained in a Wikipedia paragraph.
    """ 
    facts: List[str]

    def __len__(self):
        return len(self.facts)
    
    # make it iterable
    def __iter__(self):
        return iter(self.facts)

def load_tsvetshop_client():
    config = dotenv_values(".env")
    key = config['TSVETSHOP_KEY']
    client = openai.AzureOpenAI(
        azure_endpoint="https://tsvetshop.openai.azure.com/",
        api_key=key,
        api_version="2023-05-15"
    )
    return client

def construct_fact_decomp_prompt(src_lang, paragraph: str):
    if src_lang == 'en':
        input_prompt = f"Please breakdown the following paragraph into a list of independent facts. All of the facts should be placed in a stringified python list.\n {paragraph}"
    elif src_lang == 'fr':
        input_prompt = f"Veuillez décomposer le paragraphe suivant en une liste de faits indépendants. Tous les faits doivent être placés dans une liste python sous forme de chaîne de caractères.\n {paragraph}"
    elif src_lang == 'ru':
        input_prompt = f"Пожалуйста, разбейте следующий абзац на список независимых фактов. Все факты должны быть помещены в строковый список Python (e.g., ['Тим вырос в городе Мальорке, штат Алабама.','Его отец был работником верфи.', 'Мать Тима была домохозяйкой.','Кук получил степень бакалавра в области промышленного производства в университете Обёрна в 1982 году.','Кук получил диплом МВА в школе Фукуа университета Дьюка в 1988 году.']).\n {paragraph}"
    else: 
        raise ValueError(f"Invalid language: {src_lang}")
    return input_prompt

def ask_gpt_for_facts(client, 
                      model_name: str,
                      paragraph: str, lang_code: str):
    # message=[{"role": "user", "content": f"Please breakdown the following sentence into independent facts. Return: {sentence}"}]
    if lang_code == 'en':
        message=[{"role": "user", "content": f"Please breakdown the following paragraph into a list of independent facts. All of the facts should be placed in a stringified python list.\n {paragraph}"}]
    elif lang_code == 'fr':
        message = [{"role": "user", "content": f"Veuillez décomposer le paragraphe suivant en une liste de faits indépendants. Tous les faits doivent être placés dans une liste python sous forme de chaîne de caractères.\n {paragraph}"}]
    elif lang_code == 'ru':
        message = [{"role": "user", "content": f"Пожалуйста, разбейте следующий абзац на список независимых фактов. Все факты должны быть помещены в строковый список Python (e.g., ['Тим вырос в городе Мальорке, штат Алабама.','Его отец был работником верфи.', 'Мать Тима была домохозяйкой.','Кук получил степень бакалавра в области промышленного производства в университете Обёрна в 1982 году.','Кук получил диплом МВА в школе Фукуа университета Дьюка в 1988 году.']).\n {paragraph}"}]

    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        max_tokens = len(paragraph) + 100,
        messages = message)
    # response = client.chat.completions.create(
    #     model="gpt-4",
    #     max_tokens=len(paragraph) + 100,
    #     temperature=0,
    # messages = message)
    response_content = response.choices[0].message.content
    response_total_tokens = response.usage.total_tokens
    return response_content, response_total_tokens

def construct_fact_intersection_prompt(src_lang: str, tgt_lang_code: str, src_fact_context: List[str], 
                                       tgt_fact_context: List[str], person_name: str):
    en_tgt_lang_code_to_lang = {'en': 'English', 'fr': 'French', 'ru': 'Russian'}
    ru_lang_code_to_lang = {'en': 'английский', 'fr': 'французский', 'ru': 'русский'}
    if src_lang == 'en':
        input_prompt = f"Consider these English facts about {person_name}:\n {src_fact_context}. Is the last fact in the list inferrerable from the following {en_tgt_lang_code_to_lang[tgt_lang_code]} facts?\n {tgt_fact_context}. Return either 'yes' or 'no'."
    elif src_lang == 'fr':
        input_prompt = f"Considérez ces faits français sur {person_name}:\n {src_fact_context}. Est le dernier fait de la liste inférable de l'une des listes de faits suivantes?\n {tgt_fact_context}. Retournez 'oui' ou 'non'."
    elif src_lang == 'ru':
        input_prompt = f"Рассмотрим эти факты на русском языке о {person_name}:\n {src_fact_context}. Можно ли вывести последний факт из одного из следующих списков фактов?\n {tgt_fact_context} Возвращает список, содержащий ['да' или 'нет'] — один ответ для каждого списка фактов {ru_lang_code_to_lang[tgt_lang_code]}. Все ответы «да/нет» должны быть помещены в список строк Python. (например, ['да', 'нет', 'да'])"
    # TODO: add the Russian version of the prompt.
    return input_prompt

def ask_gpt_about_fact_intersection(client, model_name, cache,  
                                    src_lang_code, 
                                    tgt_lang_code,
                                    src_fact_context: List[str], tgt_fact_context: List[List[str]], 
                                    person_name: str, 
                                    tgt_person_name: str):
    en_lang_code_to_lang = {'en': 'English', 'fr': 'French', 'ru': 'Russian'}
    fr_lang_code_to_lang = {'en': 'anglais', 'fr': 'français', 'ru': 'russe'}
    ru_lang_code_to_lang = {'en': 'английский', 'fr': 'французский', 'ru': 'русский'}
    if src_lang_code == 'en':
        # input_prompt = f"Consider these English facts about {person_name}:\n {src_fact_context}. Is the last fact in the list inferrerable from the following French facts?\n {tgt_fact_context}. Return either 'yes' or 'no'."
        input_prompt = f"Consider these English facts about {person_name}:\n {src_fact_context}. Is the last fact in the former list inferrable from any of these lists of facts?\n {tgt_fact_context} Return a list containing ['yes' or 'no'] -- one response for each list of {en_lang_code_to_lang[tgt_lang_code]} facts. All of the yes/no responses should be placed in a stringified python list. (e.g., ['yes', 'no', 'yes'])"
    elif src_lang_code == 'fr':
        input_prompt = f"Considérez ces faits français sur {person_name}:\n {src_fact_context}. Est le dernier fait de la liste inférable de l'une des listes de faits suivantes?\n {tgt_fact_context} Retournez une liste contenant ['yes' ou 'no'] -- une réponse pour chaque liste de faits {fr_lang_code_to_lang[tgt_lang_code]}. Toutes les réponses yes/no doivent être placées dans une liste python. (e.g., ['yes', 'no', 'yes'])"
    elif src_lang_code == 'ru':
        input_prompt = f"Рассмотрим эти факты на русском языке о {tgt_person_name}:\n {src_fact_context}. Можно ли вывести последний факт из одного из следующих списков фактов?\n {tgt_fact_context} Возвращает список, содержащий ['да' или 'нет'] — один ответ для каждого списка фактов {ru_lang_code_to_lang[tgt_lang_code]}. Все ответы «да/нет» должны быть помещены в список строк Python. (например, ['да', 'нет', 'да'])"
    else:
        raise ValueError(f"Invalid language code: {src_lang_code}")
    if input_prompt not in cache:
        message=[{"role": "user", "content": input_prompt}]
        # TODO: implement this.
        def start_debug():
            logger.error("Model name is not in the list of valid models.")
            ipdb.set_trace()
        assert model_name in ['gpt4v', 'gpt-4', 'gpt-4o', 'gpt-3.5-turbo-0125'], start_debug()
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=len(src_fact_context) + len(tgt_fact_context) + 100,
            temperature=0,
        messages = message)
        # .choices[0].message.content
        response_content = response.choices[0].message.content
        if src_lang_code == 'ru':
            response_content = response_content.replace('да', 'yes').replace('нет', 'no')
        response_total_tokens = response.usage.total_tokens
        cache[input_prompt] = response_content
        return input_prompt, response_content, response_total_tokens
    else:
        return input_prompt, cache[input_prompt], 0 

def ask_gpt_if_premise_entails_hypothesis(premise: str, hypothesis: str):
    message=[{"role": "user", "content": f"Please determine if the following premise entails the following hypothesis. Return either 'entails', 'contradicts', or 'neither'.\n Premise: {premise}\n Hypothesis: {hypothesis}"}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        max_tokens=len(premise) + len(hypothesis) + 100,
        temperature=0.8,
    messages = message)
    return response

def ask_gpt_about_caa_classification(client, content: str, person_name: str ):
    # "{text}. Is the sentiment towards {word} in the given sentence positive, neutral or negative?"
    # "{text}. Is the implied sentiment towards {word} in the given text positive, neutral or negative?"
    # "{text}. Is the tone towards {word} in the given text positive, neutral or negative?"
    ## NOTE: display vs. imply
    prompt_implied_sent = f"Consider the following text: \n {content}\n. Does this content imply a positive, neutral, or negative sentiment towards {person_name}? (pos/neutral/neg/none)"
    message = [{"role": "user", "content": prompt_implied_sent}]
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages = message).choices[0].message.content
    if response not in ['pos', 'neutral', 'neg', 'none']:
        logger.warning(f"Invalid response from GPT-3: [[{response}]] for prompt:\n\n {prompt_implied_sent}")
    return response

def ask_gpt_about_caa_classification_coreference_resolution(client, content: str, person_name: str, pronoun: str):
    coref_string = f"{person_name} identifies as {pronoun}. In case of ambiguity, the pronoun {pronoun} refers to {person_name}."
    prompt_implied_sent = f"Consider the following text: \n {content}\n. {coref_string} Does this content imply a positive, neutral, or negative sentiment towards {person_name}? (pos/neutral/neg/none)"
    message = [{"role": "user", "content": prompt_implied_sent}]

    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages = message).choices[0].message.content
    if response not in ['pos', 'neutral', 'neg', 'none']:
        logger.warning(f"Invalid response from GPT-3: [[{response}]] for prompt:\n\n {prompt_implied_sent}")
    return response

def prep_caa_prompt(language, content: List[str], pronoun, person_name) -> str:
    if language == 'en':
        coref_string = f"{person_name} identifies as {pronoun}. In case of ambiguity, the pronoun {pronoun} refers to {person_name}."
        prompt_implied_sent = f"Consider the last sentence in the following text: \n {content}\n. {coref_string} Does this content imply a positive, neutral, or negative sentiment towards {person_name}? (pos/neutral/neg/none)"
    elif language == 'fr':
        coref_string = f"{person_name} s'identifie comme {pronoun}. En cas d'ambiguïté, le pronom {pronoun} se réfère à {person_name}."
        prompt_implied_sent = f"Considérez la dernière phrase du texte suivant: \n {content}\n. {coref_string} Est-ce que ce contenu implique un sentiment positif, neutre ou négatif envers {person_name}? (pos/neutre/neg/none)"
    else:
        raise ValueError(f"Invalid language: {language}")
    return prompt_implied_sent

def prompt_gpt_4(client, valid_labels: List[str], prompt) -> Tuple[str, int]:
    message = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        max_tokens = len(prompt) + 100,
        messages = message)
    response_content = response.choices[0].message.content
    if response_content not in valid_labels:
        logger.warning(f"Invalid response from GPT-4: [[{response_content}]] for prompt:\n\n {prompt}")
    response_total_tokens = response.usage.total_tokens
    return response_content, response_total_tokens