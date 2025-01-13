import json
import datetime
from typing import List
import openai
from tqdm import tqdm
import ipdb
import polars as pl
from nltk.tokenize import wordpunct_tokenize
from typing import Dict, List, Union
from collections import defaultdict
from functools import partial
from dotenv import dotenv_values
import loguru

from wikipedia_edit_scrape_tool import Paragraph, Header

from packages.gpt_query import ask_gpt_about_caa_classification, ask_gpt_about_caa_classification_coreference_resolution, prompt_gpt_4, prep_caa_prompt
from packages.flan_query import ask_flan_about_connotation, ask_mt5_about_connotation
from packages.constants import NUM_CONTEXT_CAA, GPT_CACHE_LOCATION, ANNOTATION_SAVE_PATH, CONNOTATION_FLAN_SAVE_DIR, MT5_CONNOTATION_MODEL_PATH
from packages.steps.info_diff_steps import load_other_client

logger = loguru.logger
config = dotenv_values(".env")
key = config["THE_KEY"] 

def load_tsvetshop_client():
    key = config["THE_KEY"] 
    client = openai.AzureOpenAI(
            azure_endpoint="https://tsvetshop.openai.azure.com/",
            api_key=key,
            api_version="2023-05-15"
        )
    return client

class InfoGapEmptyError(Exception):
    pass

def step_prep_for_caa(map_en_fr_info_gaps, en_bio_id, fr_bio_id, 
                      person_name, 
                      intersection_column, **kwargs):
    en_info_gap_df, fr_info_gap_df = map_en_fr_info_gaps[0].filter(pl.col('en_bio_id') == en_bio_id), map_en_fr_info_gaps[1].filter(pl.col('fr_bio_id') == fr_bio_id)
    if len(en_info_gap_df) == 0:
        raise InfoGapEmptyError(f"en_info_gap_df is empty for {en_bio_id}")
    elif len(fr_info_gap_df) == 0:
        raise InfoGapEmptyError(f"fr_info_gap_df is empty for {fr_bio_id}")
    # NOTE: 2024-03-14: noticed there was a problem of double counting quote paragraphs. This was fixed in that day, so bios moving forward should be good.
    ipdb.set_trace()
    fr_info_gap_df = fr_info_gap_df.group_by(['aligned_sentence'])\
        .agg(pl.col('fact').explode(), pl.col(intersection_column).explode(), pl.col('fact_index').explode(), pl.col('paragraph_index').explode(), 
             pl.col('pronoun').first(), pl.col('person_name').first())
    en_info_gap_df = en_info_gap_df.group_by(['aligned_sentence'])\
        .agg(pl.col('fact').explode(), pl.col(intersection_column).explode(), pl.col('paragraph_index').explode(), pl.col('fact_index').explode(),
             pl.col('pronoun').first(), pl.col('person_name').first())

    def build_caa_query_column(row, info_gap_df):
        smallest_fact_index = min(row['fact_index'])
        paragraph_indices = row['paragraph_index']
        if len(set(paragraph_indices)) > 1:
            logger.warning(f"Paragraph indices are not all the same for {person_name}: {paragraph_indices}")
            
        paragraph_index = min(paragraph_indices) # in 99.99% of cases, all of the elements will be the same
        context_lst = info_gap_df.filter((pl.col('paragraph_index').list.contains(paragraph_index)) & 
                                    (pl.col('fact_index').list.max() < smallest_fact_index))['aligned_sentence'].to_list()[-NUM_CONTEXT_CAA:]
        return context_lst + [row['aligned_sentence']]

    fr_info_gap_df = fr_info_gap_df.with_columns([
        pl.struct(['paragraph_index', 'fact_index', 'aligned_sentence'])\
            .map_elements(lambda row:
                                    build_caa_query_column(row, fr_info_gap_df)).alias('caa_query_content')
    ])
    en_info_gap_df = en_info_gap_df.with_columns([
        pl.struct(['paragraph_index', 'fact_index', 'aligned_sentence']).map_elements(lambda row:
                                                                                                build_caa_query_column(row, en_info_gap_df)).alias('caa_query_content')
    ])
    columns = ['fact', intersection_column, 'fact_index', 'paragraph_index', 'aligned_sentence', 'person_name', 'caa_query_content', 'pronoun']
    return en_info_gap_df.select(columns), fr_info_gap_df.select(columns)

def step_prep_for_caa_en_tgt(map_en_tgt_info_gaps, en_bio_id, 
                      tgt_lang_code: str,
                      tgt_bio_id, 
                      person_name, 
                      intersection_column, **kwargs):
    en_info_gap_df, tgt_info_gap_df = map_en_tgt_info_gaps[0].filter(pl.col('en_bio_id') == en_bio_id), map_en_tgt_info_gaps[1].filter(pl.col(f'{tgt_lang_code}_bio_id') == tgt_bio_id)
    if len(en_info_gap_df) == 0:
        raise InfoGapEmptyError(f"en_info_gap_df is empty for {en_bio_id}")
    elif len(tgt_info_gap_df) == 0:
        raise InfoGapEmptyError(f"fr_info_gap_df is empty for {tgt_bio_id}")
    # NOTE: 2024-03-14: noticed there was a problem of double counting quote paragraphs. This was fixed in that day, so bios moving forward should be good.
    # TODO: what happens to the person name 
    tgt_info_gap_df = tgt_info_gap_df.group_by(['aligned_sentence'])\
        .agg(pl.col('fact').explode(), pl.col(intersection_column).explode(), pl.col('fact_index').explode(), pl.col('paragraph_index').explode(), 
             pl.col('pronoun').first(), pl.col('person_name').first(), pl.col(f'{tgt_lang_code}_person_name').first())
    en_info_gap_df = en_info_gap_df.group_by(['aligned_sentence'])\
        .agg(pl.col('fact').explode(), pl.col(intersection_column).explode(), pl.col('paragraph_index').explode(), pl.col('fact_index').explode(),
             pl.col('pronoun').first(), pl.col('person_name').first(), pl.col(f'{tgt_lang_code}_person_name').first())

    def build_caa_query_column(row, info_gap_df):
        smallest_fact_index = min(row['fact_index'])
        paragraph_indices = row['paragraph_index']
        if len(set(paragraph_indices)) > 1:
            logger.warning(f"Paragraph indices are not all the same for {person_name}: {paragraph_indices}")
            
        paragraph_index = min(paragraph_indices) # in 99.99% of cases, all of the elements will be the same
        context_lst = info_gap_df.filter((pl.col('paragraph_index').list.contains(paragraph_index)) & 
                                    (pl.col('fact_index').list.max() < smallest_fact_index))['aligned_sentence'].to_list()[-NUM_CONTEXT_CAA:]
        return context_lst + [row['aligned_sentence']]

    tgt_info_gap_df = tgt_info_gap_df.with_columns([
        pl.struct(['paragraph_index', 'fact_index', 'aligned_sentence'])\
            .map_elements(lambda row:
                                    build_caa_query_column(row, tgt_info_gap_df)).alias('caa_query_content')
    ])
    en_info_gap_df = en_info_gap_df.with_columns([
        pl.struct(['paragraph_index', 'fact_index', 'aligned_sentence']).map_elements(lambda row:
                                                                                                build_caa_query_column(row, en_info_gap_df)).alias('caa_query_content')
    ])
    columns = ['fact', intersection_column, 'fact_index', 'paragraph_index', 'aligned_sentence', 'person_name', f'{tgt_lang_code}_person_name', 'caa_query_content', 'pronoun']
    return en_info_gap_df.select(columns), tgt_info_gap_df.select(columns)

def check_gpt_connotation_cache(lang_code) -> Dict[str, str]:
    cache_dir = GPT_CACHE_LOCATION
    try:
        cache_file = f"{cache_dir}/{lang_code}_connotation.json"
        with open(cache_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return defaultdict(dict)

def write_gpt_connotation_cache(lang_code, connotation_cache):
    cache_dir = GPT_CACHE_LOCATION
    cache_file = f"{cache_dir}/{lang_code}_connotation.json"
    with open(cache_file, 'w') as f:
        json.dump(connotation_cache, f)

# def step_caa_multi_sentence(en_fr_info_gaps, lgbt_paragraphs, **kwargs):

# TODO: need to replace with the new version of the function from ipynb
def step_caa_multi_sentence(en_fr_info_gaps, **kwargs):
    en_info_gap, fr_info_gap = en_fr_info_gaps
    key = config["THE_KEY"] 

    # partial(ask_gpt_for_facts, OpenAI(api_key=key))
    def get_gpt_connotation_labels(info_gap_df, lang_code):
        progress = tqdm(total=len(info_gap_df))
        if lang_code == "en":
            valid_labels = ['pos', 'neg', 'neutral', 'none']
        elif lang_code == "fr":
            valid_labels = ['pos', 'neg', 'neutre', 'none']
        partial_ask_gpt_about_caa_classification = partial(prompt_gpt_4, client=load_other_client(), valid_labels=valid_labels)
        connotation_cache = check_gpt_connotation_cache(lang_code)

        def progress_ask_gpt_about_caa_classification(query_content, person_name, pronoun):
            caa_prompt = prep_caa_prompt(lang_code, query_content, pronoun, person_name)
            if caa_prompt not in connotation_cache[person_name]:
                result, total_tokens = partial_ask_gpt_about_caa_classification(prompt=caa_prompt)
                logger.info(f"Total tokens: {total_tokens}")
                connotation_cache[person_name][caa_prompt] = result
            else:
                result = connotation_cache[person_name][caa_prompt]
            progress.update(1)
            return result 
        
        info_gap_df = info_gap_df.with_columns([
            pl.struct(['caa_query_content', 'person_name', 'pronoun']).map_elements(
                lambda row: 
                    progress_ask_gpt_about_caa_classification(
                        row['caa_query_content'], row['person_name'], row['pronoun']
                    )).alias('caa_classification')
        ])
        write_gpt_connotation_cache(lang_code, connotation_cache)
        return info_gap_df 
    en_info_gap = get_gpt_connotation_labels(en_info_gap, "en")
    fr_info_gap = get_gpt_connotation_labels(fr_info_gap, "fr")

    # concatenate the two dataframes by adding a language column
    en_info_gap = en_info_gap.with_columns([pl.lit('en').alias('language')])
    fr_info_gap = fr_info_gap.with_columns([pl.lit('fr').alias('language')])
    info_gap = pl.concat([en_info_gap, fr_info_gap])
    date = datetime.datetime.now().strftime("%m-%d")
    info_gap.write_json(f"{ANNOTATION_SAVE_PATH}/connotation_en-fr_{date}.json")
    logger.info(f"Saved the connotation data to {ANNOTATION_SAVE_PATH}/connotation_en-fr_{date}.json")
    return en_info_gap, fr_info_gap

def step_caa_multi_sentence_en_tgt(en_tgt_info_gaps, tgt_lang_code, **kwargs):
    en_info_gap, tgt_info_gap = en_tgt_info_gaps

    # partial(ask_gpt_for_facts, OpenAI(api_key=key))
    def get_gpt_connotation_labels(info_gap_df, lang_code):
        progress = tqdm(total=len(info_gap_df))
        if lang_code == "en":
            valid_labels = ['pos', 'neg', 'neutral', 'none']
        elif lang_code == "fr":
            valid_labels = ['pos', 'neg', 'neutre', 'none']
        elif lang_code == 'ru':
            # write the valid labels for Russian in Russian
            valid_labels = ['положительный', 'негативный', 'нейтральный', 'нет']
        partial_ask_gpt_about_caa_classification = partial(prompt_gpt_4, 
                                                           client=load_other_client(), 
                                                           valid_labels=valid_labels)
        connotation_cache = check_gpt_connotation_cache(lang_code)
        if not isinstance(connotation_cache, defaultdict):
            connotation_cache = defaultdict(dict, connotation_cache)

        def progress_ask_gpt_about_caa_classification(query_content, person_name, pronoun):
            if lang_code == 'ru':
                caa_prompt =  construct_prompt_2_ru(query_content, person_name, pronoun)
            elif lang_code == 'en':
                caa_prompt =  construct_prompt_2_en(query_content, person_name, pronoun)
            elif lang_code == 'fr':
                caa_prompt =  construct_prompt_2_fr(query_content,  person_name, pronoun)
            else:
                raise ValueError(f"Invalid language code: {lang_code}")
            # TODO: check that the prompt is correct
            if caa_prompt not in connotation_cache[person_name]:
                try:
                    result, total_tokens = partial_ask_gpt_about_caa_classification(prompt=caa_prompt)
                    logger.info(f"Total tokens: {total_tokens}")
                    connotation_cache[person_name][caa_prompt] = result
                except openai.BadRequestError:
                    logger.warning(f"Hit content filter with prompt: {caa_prompt}")
                    progress.update(1)
                    return 'blocked by content filter'
            else:
                result = connotation_cache[person_name][caa_prompt]
            progress.update(1)
            return result 
        person_name_column = (f'{lang_code}_person_name' if lang_code not in ['en', 'fr'] else 'person_name')
        info_gap_df = info_gap_df.with_columns([
            pl.struct(['caa_query_content', person_name_column, 'pronoun']).map_elements(
                lambda row: 
                    progress_ask_gpt_about_caa_classification(
                        row['caa_query_content'], row[person_name_column], row['pronoun']
                    )).alias('caa_classification')
        ])
        write_gpt_connotation_cache(lang_code, connotation_cache)
        return info_gap_df 
    en_info_gap = get_gpt_connotation_labels(en_info_gap, "en")
    tgt_info_gap = get_gpt_connotation_labels(tgt_info_gap, tgt_lang_code)

    # concatenate the two dataframes by adding a language column
    en_info_gap = en_info_gap.with_columns([pl.lit('en').alias('language')])
    tgt_info_gap = tgt_info_gap.with_columns([pl.lit(tgt_lang_code).alias('language')])

    info_gap = pl.concat([en_info_gap, tgt_info_gap])
    date = datetime.datetime.now().strftime("%m-%d")
    info_gap.write_json(f"{ANNOTATION_SAVE_PATH}/connotation_en-{tgt_lang_code}_{date}.json")
    logger.info(f"Saved the connotation data to {ANNOTATION_SAVE_PATH}/connotation_en-{tgt_lang_code}_{date}.json")
    return en_info_gap, tgt_info_gap

def step_caa_multi_sentence_flan(en_other_info_gaps, 
                                 other_lang_code: str,
                                 **kwargs):
    en_info_gap, other_info_gap = en_other_info_gaps
    # partial(ask_gpt_for_facts, OpenAI(api_key=key))
    def get_flan_connotation_labels(info_gap_df, lang_code):
        progress = tqdm(total=len(info_gap_df))
        if lang_code == "en":
            connotation_model_path = f"{CONNOTATION_FLAN_SAVE_DIR}_twp=False/checkpoint-4200"
            ask_flan = partial(ask_flan_about_connotation, connotation_model_path) # TODO: replace with the actual path
            generate_flan_prompt, flan_predict_connotation = ask_flan(construct_prompt_2_en)
            person_name_label = 'person_name'
        elif lang_code == "fr":
            connotation_model_path = f"{CONNOTATION_FLAN_SAVE_DIR}_twp=False/checkpoint-4200"
            ask_flan = partial(ask_flan_about_connotation, connotation_model_path) # TODO: replace with the actual path
            generate_flan_prompt, flan_predict_connotation = ask_flan(construct_prompt_2_fr)
            person_name_label = 'person_name'
        elif lang_code == 'ru':
            connotation_model_path = MT5_CONNOTATION_MODEL_PATH
            ask_flan = partial(ask_mt5_about_connotation, connotation_model_path) # TODO: replace with the actual path
            generate_flan_prompt, flan_predict_connotation = ask_flan(construct_prompt_2_ru)
            person_name_label = 'ru_person_name'
        # def progress_ask_flan_about_caa_classification(query_content, person_name, pronoun, lang_code):
        #     # TODO; May want to do the decomposition here 
        #     connotation_prediction = ask_flan_complete(query_content,  person_name, pronoun, lang_code)
        #     progress.update(1)
        #     return connotation_prediction
        info_gap_df = info_gap_df.with_columns([
            pl.struct(['caa_query_content', person_name_label, 'pronoun']).map_elements(
                lambda row: 
                    generate_flan_prompt(
                        row['caa_query_content'], row[person_name_label], row['pronoun'], lang_code
                    )).alias('caa_prompt')
        ])
        caa_prompts = info_gap_df['caa_prompt'].to_list()
        batch_size = 8
        connotation_predictions = []
        for i in range(0, len(caa_prompts), batch_size):
            batch = caa_prompts[i:i+batch_size]
            connotation_predictions.extend(flan_predict_connotation(batch))
            progress.update(len(batch))
        # add the connotation predictions to the info_gap_df
        info_gap_df = info_gap_df.with_columns([pl.Series(connotation_predictions).alias('caa_classification')])
        # remove the caa prompt column
        info_gap_df = info_gap_df.drop('caa_prompt')
        return info_gap_df 
    en_info_gap = get_flan_connotation_labels(en_info_gap, "en")
    other_info_gap = get_flan_connotation_labels(other_info_gap, other_lang_code)

    # concatenate the two dataframes by adding a language column
    en_info_gap = en_info_gap.with_columns([pl.lit('en').alias('language')])
    other_info_gap = other_info_gap.with_columns([pl.lit(other_lang_code).alias('language')])
    # info_gap = pl.concat([en_info_gap, fr_info_gap])
    # date = datetime.datetime.now().strftime("%m-%d")
    # info_gap.write_json(f"{ANNOTATION_SAVE_PATH}/connotation_en-fr_{date}.json")
    # logger.info(f"Saved the connotation data to {ANNOTATION_SAVE_PATH}/connotation_en-fr_{date}.json")
    return en_info_gap, other_info_gap

def construct_prompt_2_fr(content: str, person_name: str, pronoun: str):
    if pronoun == 'she':
        pronoun = 'elle'
    else: 
        pronoun = 'il'
    return f"Le pronom du {person_name} est '{pronoun}'. Est-ce que le texte suivant au sujet de {person_name} implique un sentiment positif, neutre ou négatif envers {person_name}? Expliquez pourquoi en une phrase. Écrivez votre réponse en format JSON avec deux clés: étiquette et explication). \n {content} (pos/neutral/neg/none)"

def construct_prompt_2_en(content: str, person_name: str, pronoun: str):
    return f"The pronoun for {person_name} is '{pronoun}'. Does the following text about {person_name} imply a positive, neutral, or negative sentiment towards {person_name}? Explain why in one sentence. Write your response in JSON format with two keys: label and explanation). \n {content} (pos/neutral/neg/none)"

def construct_prompt_2_ru(content: str, person_name: str, pronoun: str):
    if pronoun == 'she':
        pronoun = 'она'
    else:
        pronoun = 'он'
    return f"Местоимение для {person_name} - '{pronoun}'. Подразумевает ли следующий текст о {person_name} положительное, нейтральное или отрицательное отношение к {person_name}? Объясните почему в одном предложении. Напишите ваш ответ в формате JSON с двумя ключами: метка и объяснение). \n {content} (положительный/нейтральный/отрицательный/нет)"


def _parse_response_t5_label(response_raw):
    return response_raw[1:response_raw.index(',')]


def _parse_response_gpt(lang_code, prompt_content, response_raw):
    try:
        start_ind = response_raw.index('{')
        end_ind = response_raw.index('}') + 1
        response_eval = eval(response_raw[start_ind:end_ind])
        if lang_code == 'fr':
            return response_eval['étiquette']
        elif lang_code == 'en':
            return response_eval['label']
        elif lang_code == 'ru':
            return response_eval['метка']
    except:
        response_eval = response_raw
        logger.warning(f"Invalid response from GPT-4: [[{response_eval}]] for prompt:\n\n {prompt_content}")
        return response_eval

def _parse_response_rationale(lang_code, prompt_content, response_raw):
    try:
        start_ind = response_raw.index('{')
        end_ind = response_raw.index('}') + 1
        response_eval = eval(response_raw[start_ind:end_ind])
        if lang_code == 'fr':
            return response_eval['explication']
        elif lang_code == 'en':
            return response_eval['explanation']
        elif lang_code == 'ru':
            return response_eval['объяснение']

    except:
        response_eval = response_raw
        logger.warning(f"Invalid response from GPT-4: [[{response_eval}]] for prompt:\n\n {prompt_content}")
        return response_eval
    
class NoPronounError(Exception):
    pass

def step_infer_pronoun(en_content_blocks: List[Union[Paragraph,Header]], **kwargs) -> str:
    paragraphs = [block for block in en_content_blocks if isinstance(block, Paragraph)]
    # count the number of times each pronoun appears in the text: he/his/him vs. she/her/hers vs. they/them/theirs
    # return the pronoun (he/she/they) according to the most common pronoun triplet
    pronoun_counts = defaultdict(int)
    for paragraph in paragraphs:
        paragraph = paragraph.clean_text.lower()
        words = wordpunct_tokenize(paragraph)
        # count the number of times each pronoun appears in the text: he/his/him vs. she/her/hers vs. they/them/theirs
        for word in words:
            if word in ['he', 'his', 'him']:
                pronoun_counts['he'] += 1
            elif word in ['she', 'her', 'hers']:
                pronoun_counts['she'] += 1
            elif word in ['they', 'them', 'theirs']:
                pronoun_counts['they'] += 1
    try:
        pronoun = max(pronoun_counts, key=pronoun_counts.get)
    except ValueError:
        raise NoPronounError("No pronouns found in the text -- is this a disambiguation page?")
    return pronoun