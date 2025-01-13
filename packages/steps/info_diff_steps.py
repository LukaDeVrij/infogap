import os
import dill
import json
import datetime
from nltk import sent_tokenize
import json
from openai import BadRequestError, OpenAI
import pandas as pd
import openai
from nltk import sent_tokenize
from torch.cuda import OutOfMemoryError
import ipdb
import numpy as np
import polars as pl
from tqdm import tqdm
from typing import List, Union, Optional, Tuple, Dict
import loguru
from nltk import sent_tokenize
from wikipedia_edit_scrape_tool import get_text, Header, Paragraph 
from functools import partial
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from dotenv import dotenv_values

from packages.constants import LGBT_EN_WORDS, LGBT_FR_WORDS, TARGET_LANGUAGES, HF_CACHE_DIR, GPT_CACHE_LOCATION, NUM_CONTEXT_SRC, NUM_RETRIEVALS, NUM_CONTEXT_TGT, FACT_DECOMP_FLAN_SAVE_DIR, BIO_SAVE_DIR,\
    CONNOTATION_FLAN_SAVE_DIR, MT5_INFO_GAP_MODEL_PATH, MT5_FACT_DECOMP_MODEL_PATH
from packages.align_paragraphs import compute_algn_strngths, FactParagraph, prune_directional_alignments, union_forced_alignment, union_pruned_alignment
from packages.align_facts import align_facts
from packages.gpt_query import ask_gpt_for_facts, FactParagraph, ask_gpt_about_fact_intersection
from packages.flan_query import generate_facts_flan, ask_flan_about_fact_intersection, generate_facts_mt5, ask_mt5_about_fact_intersection
from packages.align_facts import pairwise_fact_fact_margin_compute, obtain_hubness_measure


logger = loguru.logger
config = dotenv_values(".env")
today = datetime.date.today()
logger.add(f"logs/info_diff_{today}.log", rotation="500 MB")

def _retrieve_relevant_paragraphs(content_blocks, wiki_lang: str, 
                                  apply_content_filter: bool) -> List[Paragraph]:
    if wiki_lang == 'enwiki':
        target_words = LGBT_EN_WORDS
    elif wiki_lang == 'frwiki':
        target_words = LGBT_FR_WORDS
    else:
        raise ValueError(f"wiki_lang must be one of {TARGET_LANGUAGES}")
    relevant = []
    for block in content_blocks:
        if isinstance(block, Paragraph):
            if apply_content_filter and any([word in block.clean_text for word in target_words]):
                relevant.append(block)
            elif not apply_content_filter:
                relevant.append(block)
    return relevant

def step_retrieve_en_content_blocks(en_bio_id: str, 
                                 **kwargs) -> List[Union[Header, Paragraph]]:
    """Return all of the headers and paragraphs from the English Wikipedia page 
    for the person with the given bio id.
    """
    english_id = en_bio_id
    en_link = f"https://en.wikipedia.org/wiki/{english_id}"
    content_blocks = get_text(en_link, 'enwiki')
    # filter out paragraphs where the clean_text attribute string has fewer than 6 words.
    content_blocks = list(filter(lambda x: not (isinstance(x, Paragraph) and len(x.clean_text.split()) < 6), content_blocks))
    return content_blocks 

class BioFilenotFoundError(Exception):
    pass

def step_retrieve_prescraped_en_content_blocks(en_bio_id, 
                                              save_dir = BIO_SAVE_DIR,
                                               **kwargs):
    try:
        with open(f"{save_dir}/{en_bio_id}_en.pkl", 'rb') as f:
            return dill.load(f)
    except FileNotFoundError:
        raise BioFilenotFoundError(f"Could not find the prescraped bio file for {en_bio_id}")

def step_retrieve_prescraped_ru_content_blocks(tgt_bio_id, **kwargs):
    try: 
        with open(f"{BIO_SAVE_DIR}/{tgt_bio_id}_ru.pkl", 'rb') as f:
            return dill.load(f)
    except FileNotFoundError:
        raise BioFilenotFoundError(f"Could not find the prescraped bio file for {tgt_bio_id}")
    
def step_retrieve_prescraped_fr_content_blocks(fr_bio_id, 
                                               save_dir = BIO_SAVE_DIR,
                                               **kwargs):
    try:
        with open(f"{save_dir}/{fr_bio_id}_fr.pkl", 'rb') as f:
            return dill.load(f)
    except FileNotFoundError:
        raise BioFilenotFoundError(f"Could not find the prescraped bio file for {fr_bio_id}")

def remove_person_specific_blocks_fr(fr_bio_id, content_blocks):
    if fr_bio_id == 'Abdellah_Taïa':
        # filter out paragraph after "Sur quelques ouvrages > La vie lente"
        sur_quelques_ouvrages_header_index = [i for i, block in enumerate(content_blocks) if isinstance(block, Header) and block.text == "Sur quelques ouvrages"][0] 
        return content_blocks[:sur_quelques_ouvrages_header_index]
    elif fr_bio_id == 'Frédéric_Mitterrand':
        s = 'prix et récompenses suivants'
        # filter out paragraph including and after "prix et récompenses suivants"
        prix_et_recompenses_index = [i for i, block in enumerate(content_blocks) if isinstance(block, Paragraph) and s in block.clean_text][0]
        return content_blocks[:prix_et_recompenses_index]
    else:
        return content_blocks

def step_retrieve_fr_content_blocks(fr_bio_id: str, 
                                 **kwargs) -> List[Union[Header, Paragraph]]:
    french_id = fr_bio_id
    fr_link = f"https://fr.wikipedia.org/wiki/{french_id}"
    content_blocks = get_text(fr_link, 'frwiki')
    num_blocks_orig = len(content_blocks)
    # voir_aussi_header = next(filter(lambda x: isinstance(x, Header) and x.text == "Voir aussi" , content_blocks))
    # check if content blocks has voir aussi header
    if any([isinstance(block, Header) and block.text == "Voir aussi" for block in content_blocks]):
        voir_aussi_header = next(filter(lambda x: isinstance(x, Header) and x.text == "Voir aussi" , content_blocks))
        voir_aussi_index = content_blocks.index(voir_aussi_header)
        content_blocks = content_blocks[:voir_aussi_index]
        # log that we're omitting the "Voir aussi" section and everything after it
        logger.info(f"Omitting the 'Voir aussi' section and everything after it for the French Wikipedia page for {fr_bio_id}. This drops {num_blocks_orig - len(content_blocks)} blocks")
    content_blocks = remove_person_specific_blocks_fr(fr_bio_id, content_blocks)
    content_blocks = list(filter(lambda x: not (isinstance(x, Paragraph) and len(x.clean_text.split()) < 6), content_blocks)) # filter out paragraphs where the clean_text attribute string has fewer than 6 words.
    return content_blocks

def step_retrieve_ru_content_blocks(ru_bio_id: str, 
                                    **kwargs): 
    ru_bio_id = ru_bio_id
    ru_link = f"https://ru.wikipedia.org/wiki/{ru_bio_id}"
    content_blocks = get_text(ru_link, 'ruwiki')
    return content_blocks

def check_gpt_fact_intersection_cache(model_name, person_name, lang_code) -> Dict[str, str]:
    cache_dir = GPT_CACHE_LOCATION
    if model_name == 'gpt-4':
        try: 
            cache_file = f"{cache_dir}/{person_name}_{lang_code}_fact_intersection.json"
            with open(cache_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    elif model_name == 'gpt4v':
        try:
            cache_file = f"{cache_dir}/{person_name}_{lang_code}_fact_intersection_gpt4v.json"
            with open(cache_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    elif model_name == 'gpt-3.5-turbo-0125':
        try:
            cache_file = f"{cache_dir}/{person_name}_{lang_code}_fact_intersection_gpt3.5_turbo_0125.json"
            with open(cache_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    elif model_name == 'gpt-4o':
        try: 
            cache_file = f"{cache_dir}/{person_name}_{lang_code}_fact_intersection_gpt4o.json"
            with open(cache_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    else:
        raise ValueError(f"model_name must be one of ['gpt-4', 'gpt-3.5']")

def write_gpt_fact_intersection_cache(model_name, person_name, lang_code, fact_cache):
    cache_dir = GPT_CACHE_LOCATION
    if model_name == 'gpt-4':
        cache_file = f"{cache_dir}/{person_name}_{lang_code}_fact_intersection.json"
    elif model_name == 'gpt-3.5-turbo-0125':
        cache_file = f"{cache_dir}/{person_name}_{lang_code}_fact_intersection_gpt3.5_turbo_0125.json"
    elif model_name == 'gpt4v':
        cache_file = f"{cache_dir}/{person_name}_{lang_code}_fact_intersection_gpt4v.json"
    elif model_name == 'gpt-4o':
        cache_file = f"{cache_dir}/{person_name}_{lang_code}_fact_intersection_gpt4o.json"
    else:
        raise ValueError(f"model_name must be one of ['gpt-4', 'gpt-3.5']")
    with open(cache_file, 'w') as f:
        json.dump(fact_cache, f)

def check_gpt_fact_cache(person_name, lang_code) -> Dict[str, str]:
    cache_dir = GPT_CACHE_LOCATION
    try:
        cache_file = f"{cache_dir}/{person_name}_{lang_code}_facts.json"
        with open(cache_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def write_gpt_fact_cache(person_name, lang_code, fact_cache):
    cache_dir = GPT_CACHE_LOCATION
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{person_name}_{lang_code}_facts.json"
    with open(cache_file, 'w') as f:
        json.dump(fact_cache, f)

def load_other_client():
    key = config['THE_KEY']
    client = OpenAI(api_key=key)  # TODO: put this in an env instead.
    return client

def extract_fact_decomp_list(response: str) -> List[str]:
    if '```' in response:
        # return everything between the first occurrence of '[' and the last occurrence of ']'
        start_index = response.index('[')
        end_index = response.rindex(']')
        return response[start_index:end_index + 1]
    else:
        return response

def step_generate_facts(content_blocks: List[Union[Paragraph,Header]], 
                        lang_code: str,
                        person_name: str,
                        model_name: str = 'gpt-4o',
                        **kwargs) -> List[List[str]]:
    # TODO: need to update this to filter out headers.
    all_facts = []
    paragraphs = [block for block in content_blocks if isinstance(block, Paragraph)]
    # client = load_tsvetshop_client() 
    client = load_other_client()
    ask_for_facts = partial(ask_gpt_for_facts, client, model_name)
    fact_cache = check_gpt_fact_cache(person_name, lang_code)
    total_num_tokens = 0
    for paragraph in tqdm(paragraphs):
        if paragraph.clean_text in fact_cache :
            all_facts.append(FactParagraph(fact_cache[paragraph.clean_text]))
            continue
        elif paragraph.clean_text + "    *: premier" in fact_cache:
            all_facts.append(FactParagraph(fact_cache[paragraph.clean_text + "    *: premier"]))
        else:
            try:
                response, num_tokens = ask_for_facts(paragraph.clean_text, lang_code)
            except BadRequestError as e:
                sentences = sent_tokenize(paragraph.clean_text)
                # get the error message from the exception
                error_message = e.args[0]
                logger.error(f"Content warning from openai for paragraph: {paragraph.clean_text}. The error message is: {error_message}. Used sentence tokenization instead; there are {len(sentences)} sentences.")
                all_facts.append(FactParagraph(sentences))
                continue
            try:
                fact_str = extract_fact_decomp_list(response) 
                fact_list = list(eval(fact_str))
                # NOTE: this if the prompt is changed, the cache will *not* be updated. something to keep in mind.
                fact_cache[paragraph.clean_text] = fact_list
                all_facts.append(FactParagraph(fact_list))
            except:
                ipdb.set_trace()
                sentences = sent_tokenize(paragraph.clean_text)
                logger.warning(f"Could not parse facts from paragraph: {paragraph.clean_text}. Used sentence tokenization instead; there are {len(sentences)} sentences.")
                all_facts.append(FactParagraph(sentences))
            # except:
            #     logger.error(f"Could not parse facts from paragraph: {paragraph.clean_text}")
            #     raise ValueError(f"Could not parse facts from paragraph: {paragraph.clean_text}")
            total_num_tokens += num_tokens
    assert len(all_facts) == len(paragraphs)
    # log the number of tokens required to generate the facts for the person
    write_gpt_fact_cache(person_name, lang_code, fact_cache)
    logger.info(f"Total number of tokens required to generate the facts for {person_name} in {lang_code}: {total_num_tokens}")
    # write the fact cache to a file
    return all_facts

def _get_non_current_paragraph_facts(paragraph_index, all_facts: List[FactParagraph]):
    non_current_paragraphs = []
    for i, fact_paragraph in enumerate(all_facts):
        if i != paragraph_index:
            non_current_paragraphs.extend(fact_paragraph.facts)
    return non_current_paragraphs

def step_get_tgt_retrieval_candidates(all_en_fact_blocks: List[FactParagraph],
                                      all_tgt_fact_blocks: List[FactParagraph],
                                      annotation_frame: pl.DataFrame,
                                      person_name: str,
                                      **kwargs): 
    def _get_query_facts(frame):
        # TODO: implement this
        src_contexts = frame['src_context'].to_list()
        query_facts = [src_context[-1] for src_context in src_contexts]
        return pl.DataFrame({
            'fact': query_facts, 
            'fact_embed': [model.encode(fact) for fact in query_facts]
        })
    
    en_src_facts_frame = annotation_frame.filter(pl.col('language')=='en').select(['src_context', 'paragraph_index'])
    fr_src_facts_frame = annotation_frame.filter(pl.col('language')=='fr').select(['src_context', 'paragraph_index'])

    def get_tgt_contexts_from_margins(margins: np.array, tgt_facts: List[str], 
                         k=2) -> List[List[List[str]]]:
        assert margins.shape[1] == len(tgt_facts)
        # get the top k indices for each row from the margins matrix.
        # the margins represent similarity value, so higher values are better.
        # we then want to use these indices to get the corresponding facts from the tgt_facts list.

        # for each of those facts, we'll also get the two facts preceding it.
        # we'll return a list of lists of strings, where each inner list contains the three facts.
        # (note that if the index is 0 or 1, we'll just return the first or first two facts in the list)
        top_k_indices = np.argsort(margins, axis=1)[:, -k:]
        all_top_k_facts = []

        for i in range(margins.shape[0]):
            top_k_facts = []
            for j in range(k):
                tgt_facts_index = top_k_indices[i, j]
                top_k_facts.append(tgt_facts[max(0, tgt_facts_index - 2):tgt_facts_index + 1])
            all_top_k_facts.append(top_k_facts)
        return all_top_k_facts
    
    model = SentenceTransformer('sentence-transformers/LaBSE', cache_folder=HF_CACHE_DIR)
    def get_tgt_contexts(src_query_df, src_fact_blocks, tgt_fact_blocks):
        all_src_facts  = [fact for fact_paragraph in src_fact_blocks for fact in fact_paragraph.facts]
        all_tgt_facts = [fact for fact_paragraph in tgt_fact_blocks for fact in fact_paragraph.facts]
        tgt_fact_frame = pl.DataFrame({
            'fact': all_tgt_facts,
            'fact_embed': [model.encode(fact) for fact in all_tgt_facts]
        })
        # Just do En first.
        try:
            en_query_facts_df = obtain_hubness_measure(_get_query_facts(src_query_df), all_src_facts + [person_name]) # add hubness_cosine column
            tgt_all_facts_df = obtain_hubness_measure(tgt_fact_frame, all_tgt_facts + [person_name]) # add hubness_cosine column
        except np.AxisError:
            ipdb.set_trace()

        # lang_two_facts_df = obtain_hubness_measure(lang_two_facts_df, all_tgt_facts)
        comparison_fn = partial(pairwise_fact_fact_margin_compute, use_margin_adjustment=True)
        query_margins = comparison_fn(en_query_facts_df, tgt_all_facts_df)
        tgt_contexts = get_tgt_contexts_from_margins(query_margins, all_tgt_facts)
        return tgt_contexts
    
    result_dfs = []
    if len(en_src_facts_frame) != 0: 
        fr_tgt_contexts = get_tgt_contexts(en_src_facts_frame, all_en_fact_blocks, all_tgt_fact_blocks)
        en_src_facts_frame = en_src_facts_frame.with_columns([
            pl.lit(fr_tgt_contexts).alias('tgt_contexts'),
            pl.lit('en').alias('language')
        ])
        result_dfs.append(en_src_facts_frame)
    if len(fr_src_facts_frame) != 0:
        en_tgt_contexts = get_tgt_contexts(fr_src_facts_frame, all_tgt_fact_blocks, all_en_fact_blocks)
        fr_src_facts_frame = fr_src_facts_frame.with_columns([
            pl.lit(en_tgt_contexts).alias('tgt_contexts'),
            pl.lit('fr').alias('language')
        ])
        result_dfs.append(fr_src_facts_frame)
    return pl.concat(result_dfs)

def step_generate_facts_flan(content_blocks: List[Union[Paragraph, Header]],
                             lang_code: str, 
                             person_name: str,
                             **kwargs) -> List[List[str]]:
    all_facts = []
    paragraphs = [block for block in content_blocks if isinstance(block, Paragraph)]
    if lang_code == 'en' or lang_code == 'fr':
        output_dir = f"{FACT_DECOMP_FLAN_SAVE_DIR}_twp=False/checkpoint-600" # twp: train with peft
        ask_for_facts = generate_facts_flan(output_dir, lang_code)
    elif lang_code == 'ru':
        ask_for_facts = generate_facts_mt5(MT5_FACT_DECOMP_MODEL_PATH, lang_code)
    else:
        raise ValueError(f"lang_code must be one of ['en', 'fr', 'ru']")

    progress = tqdm(total=len(paragraphs))
    # TODO: iterate two paragraphs at a time
    for i in range(0, len(paragraphs), 2):
        try:
            batch_paragraphs = paragraphs[i:i+2]
            responses = ask_for_facts([paragraph.clean_text for paragraph in batch_paragraphs])
            responses = [sent_tokenize(response) for response in responses]
            all_facts.extend([FactParagraph(response) for response in responses])
            progress.update(len(batch_paragraphs))
        except OutOfMemoryError:
            for paragraph in batch_paragraphs:
                response = ask_for_facts([paragraph.clean_text])
                response = sent_tokenize(response)
                all_facts.append(FactParagraph(response))
            progress.update(len(batch_paragraphs))
    # for paragraph in tqdm(paragraphs): # TODO: can we create two paragraphs at a time?  response = ask_for_facts(paragraph.clean_text)
    #     # response = response.split("\n") # apparently the finetuning can't generate new lines
    #     response = sent_tokenize(response)
    #     all_facts.append(FactParagraph(response))
    return all_facts

def _create_fact_df(facts: List[List[str]]) -> pl.DataFrame:
    fact_paragraph_nums = []
    for i in range(len(facts)):
        fact_paragraph_nums.extend([i for _ in range(len(facts[i]))])

    # create a polars dataframe with the en senteces with columns paragraph_num and fact
    # flatten the fact list 
    fact_df = pl.DataFrame({
        'paragraph_index': fact_paragraph_nums, 
        'fact': [fact for fact_list in facts for fact in fact_list]
    })
    return fact_df

def _create_fact_df_with_fact_index(fact_blocks: List[FactParagraph]) -> pl.DataFrame:
    fact_paragraph_nums = []
    for i in range(len(fact_blocks)):
        fact_paragraph_nums.extend([i for _ in range(len(fact_blocks[i].facts))])
    fact_indices = []
    j = 0
    for i in range(len(fact_blocks)):
        for _ in range(len(fact_blocks[i].facts)):
            fact_indices.append(j)
            j += 1
    fact_df = pl.DataFrame({
        'paragraph_index': fact_paragraph_nums,
        'fact': [fact for fact_block in fact_blocks for fact in fact_block.facts],
        'fact_index': fact_indices
    })
    return fact_df

def step_obtain_paragraphs_associations(step_name: str, version: str, 
                               en_facts: List[List[str]], fr_facts: List[List[str]], 
                               **kwargs) -> Tuple[np.array, np.array]:
    # create two polars dataframes, one for the english facts and one for the french facts
    # the columns should be 'fact' and 'paragraph_index'
    model = SentenceTransformer('sentence-transformers/LaBSE', cache_folder=HF_CACHE_DIR)
    en_fact_df = _create_fact_df(en_facts)
    fr_fact_df = _create_fact_df(fr_facts)
    # add the sentence embedding column, called 'fact_embed'
    def add_embed_column(fact_df):
        fact_df = fact_df.with_columns([
            pl.col('fact').map_elements(lambda fact: model.encode(fact)).alias('fact_embed')
        ])
        return fact_df
    en_fact_df = add_embed_column(en_fact_df)
    fr_fact_df = add_embed_column(fr_fact_df)
    en_fr_algn_strn, fr_en_algn_strn = compute_algn_strngths(en_fact_df, fr_fact_df, 'hubness_margin')
    assert en_fr_algn_strn.shape[0] == len(en_fact_df['paragraph_index'].unique())
    assert fr_en_algn_strn.shape[0] == len(fr_fact_df['paragraph_index'].unique())
    return en_fr_algn_strn, fr_en_algn_strn

def step_obtain_en_ru_paragraphs_associations(step_name: str, version: str, 
                               en_facts: List[List[str]], ru_facts: List[List[str]], 
                               **kwargs) -> Tuple[np.array, np.array]:
    # create two polars dataframes, one for the english facts and one for the french facts
    # the columns should be 'fact' and 'paragraph_index'
    model = SentenceTransformer('sentence-transformers/LaBSE', cache_folder=HF_CACHE_DIR)
    en_fact_df = _create_fact_df(en_facts)
    ru_fact_df = _create_fact_df(ru_facts)
    # add the sentence embedding column, called 'fact_embed'
    def add_embed_column(fact_df):
        fact_df = fact_df.with_columns([
            pl.col('fact').map_elements(lambda fact: model.encode(fact)).alias('fact_embed')
        ])
        return fact_df
    en_fact_df = add_embed_column(en_fact_df)
    ru_fact_df = add_embed_column(ru_fact_df)
    en_fr_algn_strn, ru_en_algn_strn = compute_algn_strngths(en_fact_df, ru_fact_df, 'hubness_margin')
    assert en_fr_algn_strn.shape[0] == len(en_fact_df['paragraph_index'].unique())
    assert ru_en_algn_strn.shape[0] == len(ru_fact_df['paragraph_index'].unique())
    return en_fr_algn_strn, ru_en_algn_strn

def step_union_alignments(unpruned_alignment_strns: Tuple[np.array, np.array], 
                          **kwargs) -> pl.DataFrame:
    unpruned_alignment_matrix = union_forced_alignment(unpruned_alignment_strns[0].argmax(axis=1), unpruned_alignment_strns[1].argmax(axis=1))
    aligned_rows = np.where(unpruned_alignment_matrix > 0)[0]
    aligned_columns = np.where(unpruned_alignment_matrix > 0)[1]
    if 'lang_code' in kwargs:
        lang_code = kwargs['lang_code']
    else:
        lang_code = 'fr'
    alignment_df = pl.DataFrame({
        'en_paragraph_index': aligned_rows,
        (f'{lang_code}_paragraph_index'): aligned_columns
    })
    return alignment_df

def _compute_info_gap(src_fact_df, tgt_fact_df, alignment_df: pl.DataFrame,
                      src_lang_code: str, tgt_lang_code: str,
                      src_person_name: str, # e.g., Tim Cook 
                      tgt_person_name): # e.g., Tim Cook for FR, or Тим Кук for RU
    info_intersection_mapping = defaultdict(list) # Dict[int, int]: key is the src fact index and value are candidate tgt facts

    tgt_fact_df = tgt_fact_df.sort('fact_index', descending=False)

    for src_paragraph_index in tqdm(src_fact_df['paragraph_index'].unique()):
        src_sub_fact_df = src_fact_df.filter(pl.col('paragraph_index') == src_paragraph_index)
        # get the aligned paragraphs for this paragraph
        aligned_tgt_paragraph_indices = alignment_df.filter(pl.col(f'{src_lang_code}_paragraph_index') == src_paragraph_index)[f'{tgt_lang_code}_paragraph_index'].to_list()
        tgt_sub_fact_df = tgt_fact_df
        # get distractor french facts: 100 facts from paragraphs that are not in aligned_tgt_paragraph_indices

        # distractor_fact_df = tgt_fact_df.filter(~pl.col('paragraph_index').is_in(aligned_tgt_paragraph_indices)).sample(100) # by default, sample without replacement. Will need to change this to use a min()
        # sample 20, or the number of facts in the aligned paragraphs, whichever is smaller
        distractor_fact_df = tgt_fact_df.filter(~pl.col('paragraph_index').is_in(aligned_tgt_paragraph_indices))
        distractor_fact_df = distractor_fact_df.sample(min(50, len(distractor_fact_df)), with_replacement=False) # TODO: we should look into the sensitivity of this hyperparameter later
        background_src_fact_df = src_fact_df.filter(~pl.col('paragraph_index').is_in([src_paragraph_index]))
        background_src_fact_df = background_src_fact_df.sample(min(50, len(background_src_fact_df)), with_replacement=False)
        margin_matrix_forward = align_facts(src_sub_fact_df, tgt_sub_fact_df, 
                                            background_src_fact_df['fact'].to_list() + [src_person_name], 
                                            distractor_fact_df['fact'].to_list() + [tgt_person_name])
        # get the en facts (rows) where there is a value greater than 0 in the margin matrix

        # TODO: this needs to be changed. Just take e.g. the top 3, instead of all the facts that are greater than 0
        # matched_facts = np.where(margin_matrix_forward > 0)
        # use argsort with k = 5 to get the top 5 aligned facts.
        top_k = min(5, len(tgt_sub_fact_df))
        matched_fact_inds = np.argsort(margin_matrix_forward, axis=1)[:, -top_k:]

        ## we can just add all the margin entries to the intersection mapping; we don't need to take just the greater than 0 entries.
        # for src_fact_index, tgt_fact_index in zip(matched_fact_inds[0], matched_fact_inds[1]):
        for i in range(matched_fact_inds.shape[0]):
            for j in range(matched_fact_inds.shape[1]):
                info_intersection_mapping[src_sub_fact_df['fact_index'][i]].append(
                    (tgt_sub_fact_df['fact_index'][int(matched_fact_inds[i,j])],
                    margin_matrix_forward[i, matched_fact_inds[i,j]])
                )
    # pl.col('fact_index').map_elements(lambda index: info_intersection_mapping[index]).alias('info_intersection_mapping')
    # use the info_intersection_mapping if the fact index is in the info_intersection_inds, otherwise set to 'not in info intersection'
    src_fact_df = src_fact_df.with_columns([
        pl.col('fact_index').map_elements(lambda index: info_intersection_mapping[index])\
            .alias('info_retrieval_mapping')
    ])
    return src_fact_df

def step_retrieve_potential_matches( en_bio_id: str, fr_bio_id: str,
                     en_facts: List[List[str]], fr_facts: List[List[str]], 
                     alignment_df: pl.DataFrame, 
                     person_name: str,
                     **kwargs) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Compute the information gap between the English and French paragraphs.
    Every row in {en_fact_df} and {fr_fact_df} will be annotated with whether it is in the:
    - strong information diff (whether it is part of a paragraph that was not aligned to any other paragraph)
    - weak information diff (whether it is part of a paragraph that was aligned to another paragraph, but wasn't aligned to a 
        fact in the aligned paragraph)
    - info intersection: whether it is part of a paragraph that was aligned to another paragraph and was also aligned to a fact
        in the aligned paragraph
    """
    # we can use the same decision rule as the one used in the alignment pruning step for paragraphs.
    model = SentenceTransformer('sentence-transformers/LaBSE', cache_folder=HF_CACHE_DIR)
    en_fact_df = _create_fact_df(en_facts)
    fr_fact_df = _create_fact_df(fr_facts)
    # add an index column to the fact dataframes
    en_fact_df = en_fact_df.with_columns([
        pl.lit(pl.Series(range(len(en_fact_df)))).alias('fact_index')
    ])
    fr_fact_df = fr_fact_df.with_columns([
        pl.lit(pl.Series(range(len(fr_fact_df)))).alias('fact_index')
    ])
    def encode_fact(progress, atomic_fact):
        progress.update(1)
        return model.encode(atomic_fact)
    # add the fact embeddings to the fact dataframes
    logger.info(f"Encoding English facts")
    progress = tqdm(total=len(en_fact_df))
    encode_fn = partial(encode_fact, progress)
    en_fact_df = en_fact_df.with_columns([
        pl.col('fact').map_elements(encode_fn).alias('fact_embed')
    ])
    progress = tqdm(total=len(fr_fact_df))
    logger.info(f"Encoding French facts")
    encode_fn = partial(encode_fact, progress)
    fr_fact_df = fr_fact_df.with_columns([
        pl.col('fact').map_elements(encode_fn).alias('fact_embed')
    ])

    en_info_gap_df = _compute_info_gap(en_fact_df, fr_fact_df, alignment_df, 'en', 'fr', person_name, person_name)
    fr_info_gap_df = _compute_info_gap(fr_fact_df, en_fact_df, alignment_df, 'fr', 'en', person_name, person_name)
    # add a literal column to the info gap dataframes, with the en_bio_id
    en_info_gap_df = en_info_gap_df.with_columns([
        pl.lit(en_bio_id).alias('en_bio_id'),
        pl.lit(person_name).alias('person_name')
    ])
    fr_info_gap_df = fr_info_gap_df.with_columns([
        pl.lit(fr_bio_id).alias('fr_bio_id'),
        pl.lit(person_name).alias('person_name')
    ])
    alignment_df = alignment_df.with_columns([
        pl.lit(en_bio_id).alias('en_bio_id'),
        pl.lit(fr_bio_id).alias('fr_bio_id'),
        pl.lit(person_name).alias('person_name')
    ])

    return en_info_gap_df.drop("fact_embed").to_pandas(), fr_info_gap_df.drop("fact_embed").to_pandas(), alignment_df.to_pandas()

# TODO: change this to call _compute_info_gap properly.
def step_retrieve_potential_matches_en_tgt( en_bio_id: str, tgt_bio_id: str,
                     en_facts: List[List[str]], tgt_facts: List[List[str]], 
                     alignment_df: pl.DataFrame, 
                     lang_code: str,
                     person_name: str,
                     tgt_person_name: str,
                     **kwargs) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Compute the information gap between the English and French paragraphs.
    Every row in {en_fact_df} and {fr_fact_df} will be annotated with whether it is in the:
    - strong information diff (whether it is part of a paragraph that was not aligned to any other paragraph)
    - weak information diff (whether it is part of a paragraph that was aligned to another paragraph, but wasn't aligned to a 
        fact in the aligned paragraph)
    - info intersection: whether it is part of a paragraph that was aligned to another paragraph and was also aligned to a fact
        in the aligned paragraph
    """
    # we can use the same decision rule as the one used in the alignment pruning step for paragraphs.
    model = SentenceTransformer('sentence-transformers/LaBSE', cache_folder=HF_CACHE_DIR)
    en_fact_df = _create_fact_df(en_facts)
    tgt_fact_df = _create_fact_df(tgt_facts)
    # add an index column to the fact dataframes
    en_fact_df = en_fact_df.with_columns([
        pl.lit(pl.Series(range(len(en_fact_df)))).alias('fact_index')
    ])
    tgt_fact_df = tgt_fact_df.with_columns([
        pl.lit(pl.Series(range(len(tgt_fact_df)))).alias('fact_index')
    ])
    # add the fact embeddings to the fact dataframes
    en_fact_df = en_fact_df.with_columns([
        pl.col('fact').map_elements(lambda fact: model.encode(fact)).alias('fact_embed')
    ])
    tgt_fact_df = tgt_fact_df.with_columns([
        pl.col('fact').map_elements(lambda fact: model.encode(fact)).alias('fact_embed')
    ])

    en_info_gap_df = _compute_info_gap(en_fact_df, tgt_fact_df, alignment_df, 'en', lang_code, person_name, tgt_person_name)
    tgt_info_gap_df = _compute_info_gap(tgt_fact_df, en_fact_df, alignment_df, lang_code, 'en', person_name, tgt_person_name)
    # add a literal column to the info gap dataframes, with the en_bio_id
    en_info_gap_df = en_info_gap_df.with_columns([
        pl.lit(en_bio_id).alias('en_bio_id'),
        pl.lit(person_name).alias('person_name'),
        pl.lit(tgt_person_name).alias(f'{lang_code}_person_name')
    ])
    tgt_info_gap_df = tgt_info_gap_df.with_columns([
        pl.lit(tgt_bio_id).alias(f'{lang_code}_bio_id'),
        pl.lit(person_name).alias('person_name'),
        pl.lit(tgt_person_name).alias(f'{lang_code}_person_name')
    ])
    alignment_df = alignment_df.with_columns([
        pl.lit(en_bio_id).alias('en_bio_id'),
        pl.lit(tgt_bio_id).alias(f'{lang_code}_bio_id'),
        pl.lit(person_name).alias('person_name'),
        pl.lit(tgt_person_name).alias(f'{lang_code}_person_name')
    ])
    return en_info_gap_df.drop("fact_embed").to_pandas(), tgt_info_gap_df.drop("fact_embed").to_pandas(), alignment_df.to_pandas()

def step_extract_person_abln(annotated_frame: pl.DataFrame, person_name: str, **kwargs):
    return annotated_frame.filter(pl.col('person_name')==person_name)

def step_compute_info_gap_reasoning(info_gap_retrieval_dfs: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], 
                                    person_name: str, 
                                    tgt_person_name: str,
                                    model_name: str,
                                    **kwargs):
    # TODO: we can adapt this to load the lang code if it's in there.
    if 'lang_code' in kwargs:
        lang_code = kwargs['lang_code']
    else:
        lang_code = 'fr'
    en_info_gap_df, tgt_info_gap_df, alignment_df = info_gap_retrieval_dfs
    en_info_gap_df = pl.from_pandas(en_info_gap_df) if isinstance(en_info_gap_df, pd.DataFrame) else en_info_gap_df
    tgt_info_gap_df = pl.from_pandas(tgt_info_gap_df) if isinstance(tgt_info_gap_df, pd.DataFrame) else tgt_info_gap_df

    assert len(en_info_gap_df['person_name'].unique()) == 1
    # client = load_tsvetshop_client()
    client = load_other_client()
    en_fact_intersection_cache = check_gpt_fact_intersection_cache(model_name, person_name, 'en')
    tgt_fact_intersection_cache = check_gpt_fact_intersection_cache(model_name, person_name, lang_code) 
    ask_about_intersection = partial(ask_gpt_about_fact_intersection, client, model_name)

    progress = tqdm(total=len(tgt_info_gap_df) + len(en_info_gap_df))
    full_info_gap_num_tokens = {'en': 0, lang_code: 0} 
    def annotate_llm(cache, src_lang, tgt_lang, person_name: str, src_info_gap_df: pl.DataFrame, tgt_info_gap_df: pl.DataFrame, 
                     paragraph_index,  info_intersection_mapping, fact_index):
        src_paragraph_index = paragraph_index
        src_fact_context = src_info_gap_df.filter((pl.col('paragraph_index') == src_paragraph_index) & (pl.col('fact_index') <= fact_index))['fact'].to_list()[-NUM_CONTEXT_SRC:]
        tgt_contexts = []
        for tgt_index, margin in list(sorted(info_intersection_mapping, key=lambda x: x[1], reverse=True))[:NUM_RETRIEVALS]:
            try:
                tgt_paragraph_index = tgt_info_gap_df.filter(pl.col('fact_index') == tgt_index)['paragraph_index'].to_list()[0]
            except IndexError:
                ipdb.set_trace()
            tgt_context = tgt_info_gap_df.filter((pl.col('paragraph_index') == tgt_paragraph_index) & (pl.col('fact_index') <= tgt_index))['fact'].to_list()[-NUM_CONTEXT_TGT:]
            tgt_contexts.append(tgt_context)
            try:
                input_prompt, response, num_tokens = ask_about_intersection(cache, src_lang, tgt_lang, src_fact_context, tgt_contexts, person_name, tgt_person_name)
                total_num_tokens = num_tokens
                full_info_gap_num_tokens[src_lang] += total_num_tokens
                if num_tokens == 0:
                    if input_prompt in cache:
                        logger.info(f"Cache hit with input prompt: {input_prompt}")
                gpt_intersection_labels = response
                # log the number of tokens required to validate intersection labels for the person
                logger.info(f"{total_num_tokens}")
                progress.update(1)
                return str(gpt_intersection_labels)
            except BadRequestError:
                progress.update(1)
                logger.warning(f"Content warning for src_fact_context: {src_fact_context} and tgt_contexts: {tgt_contexts}.")
                return 'failed due to content policy'
    annotation_fn = partial(annotate_llm, tgt_fact_intersection_cache, lang_code, 'en', tgt_info_gap_df['person_name'][0], tgt_info_gap_df, en_info_gap_df)
    tgt_info_gap_df = tgt_info_gap_df.with_columns([
        pl.struct(['paragraph_index', 'fact_index', 'info_retrieval_mapping']).\
            map_elements(lambda row: annotation_fn(row['paragraph_index'], row['info_retrieval_mapping'], row['fact_index'])).\
                alias(f'{model_name}_intersection_label')
    ]).with_columns([
        pl.lit(tgt_person_name).alias(f'{lang_code}_person_name')
    ])
    logger.info(f"Total number of tokens required to validate intersection labels for {person_name} in {lang_code}: {full_info_gap_num_tokens[lang_code]}")
    annotation_fn = partial(annotate_llm, en_fact_intersection_cache, 'en', lang_code, en_info_gap_df['person_name'][0], en_info_gap_df, tgt_info_gap_df)
    en_info_gap_df = en_info_gap_df.with_columns([
        pl.struct(['paragraph_index', 'fact_index',  'info_retrieval_mapping']).\
            map_elements(lambda row: annotation_fn(row['paragraph_index'], row['info_retrieval_mapping'], row['fact_index'])).\
                alias(f'{model_name}_intersection_label')
    ]).with_columns([
        pl.lit(tgt_person_name).alias(f'{lang_code}_person_name')
    ])
    logger.info(f"Total number of tokens required to validate intersection labels for {person_name} in en: {full_info_gap_num_tokens['en']}")
    # write the fact intersection cache to a file
    write_gpt_fact_intersection_cache(model_name, person_name, 'en', en_fact_intersection_cache)
    write_gpt_fact_intersection_cache(model_name, person_name, lang_code, tgt_fact_intersection_cache)
    return en_info_gap_df, tgt_info_gap_df, alignment_df

def step_compute_info_gap_reasoning_simple(person_name, 
                                      query_df: pl.DataFrame,
                                      **kwargs):
    client = load_other_client()
    ask_about_intersection = partial(ask_gpt_about_fact_intersection, client, "gpt-4", {})

    def assign_gpt_label(gpt_intersection_labels: str):
        # NOTE: be careful about what happens here with the 'not intersection' case
        # by accident it has worked out (assigned 'no') but it might not always work out
        return 'yes' if ('y' in [label.strip().lower() for label in gpt_intersection_labels]) else 'no'

    en_query_df = query_df.filter(pl.col('language') == 'en')
    result_dfs = []
    if len(en_query_df) != 0:
        en_query_df = en_query_df.with_columns([
            pl.struct(['src_context', 'tgt_contexts'])\
                .map_elements(lambda row: assign_gpt_label(ask_about_intersection('en', 'fr', 
                                                                row['src_context'], 
                                                                row['tgt_contexts'], 
                                                                person_name, 
                                                                person_name)[1])).alias('gpt-4_intersection_label'),
            pl.lit(person_name).alias('person_name')
        ])
        result_dfs.append(en_query_df)
    fr_query_df = query_df.filter(pl.col('language') == 'fr')
    if len(fr_query_df) != 0:
        fr_query_df = fr_query_df.with_columns([
            pl.struct(['src_context', 'tgt_contexts'])\
                .map_elements(lambda row: assign_gpt_label(ask_about_intersection('fr', 'en', 
                                                                row['src_context'], 
                                                                row['tgt_contexts'], 
                                                                person_name, 
                                                                person_name)[1])).alias('gpt-4_intersection_label'),
            pl.lit(person_name).alias('person_name')
        ])
        result_dfs.append(fr_query_df)
    return pl.concat(result_dfs)

class ExceptionOOMSingleDataPoint(Exception):
    pass

def step_compute_info_gap_reasoning_flan(info_gap_retrieval_dfs: Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame], 
                                         other_lang_code: str,
                                         model_name: str,
                                         **kwargs):
    en_info_gap_df, other_info_gap_df, alignment_df = info_gap_retrieval_dfs
    # convert to polars dataframes if they are not already
    en_info_gap_df = pl.from_pandas(en_info_gap_df) if isinstance(en_info_gap_df, pd.DataFrame) else en_info_gap_df
    other_info_gap_df = pl.from_pandas(other_info_gap_df) if isinstance(other_info_gap_df, pd.DataFrame) else other_info_gap_df
    
    progress = tqdm(total=len(other_info_gap_df) + len(en_info_gap_df))

    if other_lang_code == 'fr':
        generate_prompt, predict_local  = ask_flan_about_fact_intersection(f"{CONNOTATION_FLAN_SAVE_DIR}_info_gap_twp=False/checkpoint-8600") # TODO: need to double check the checkpoint
    elif other_lang_code == 'ru':
        # TODO: need to use the MT5 checkpoint
        generate_prompt, predict_local = ask_mt5_about_fact_intersection(MT5_INFO_GAP_MODEL_PATH)
        # generate_flan_prompt, predict_flan = ask_flan_about_fact_intersection(f"{CONNOTATION_FLAN_SAVE_DIR}_info_gap_twp=False_lang_pair=ru_en/checkpoint-8200")
    def annotate_llm(src_lang, tgt_lang_code, person_name: str, src_info_gap_df: pl.DataFrame, tgt_info_gap_df: pl.DataFrame, 
                    paragraph_index,  info_intersection_mapping, fact_index):
        src_paragraph_index = paragraph_index
        src_fact_context = src_info_gap_df.filter((pl.col('paragraph_index') == src_paragraph_index) & (pl.col('fact_index') <= fact_index))['fact'].to_list()[-NUM_CONTEXT_SRC:]
        tgt_contexts = []
        for tgt_index, margin in list(sorted(info_intersection_mapping, key=lambda x: x[1], reverse=True))[:NUM_RETRIEVALS]:
            try:
                tgt_paragraph_index = tgt_info_gap_df.filter(pl.col('fact_index') == tgt_index)['paragraph_index'].to_list()[0]
            except IndexError:
                ipdb.set_trace()
            tgt_context = tgt_info_gap_df.filter((pl.col('paragraph_index') == tgt_paragraph_index) & (pl.col('fact_index') <= tgt_index))['fact'].to_list()[-NUM_CONTEXT_TGT:]
            tgt_contexts.append(tgt_context)
        response = generate_prompt(src_lang, tgt_lang_code, 
                                               src_fact_context, 
                                               tgt_contexts, 
                                               person_name)
        return str(response)

    annotation_fn = partial(annotate_llm, other_lang_code, 'en', other_info_gap_df[f'{other_lang_code}_person_name'][0], other_info_gap_df, en_info_gap_df)
    other_intersection_labels = []
    other_info_gap_df = other_info_gap_df.with_columns([
        pl.struct(['paragraph_index', 'fact_index', 'info_retrieval_mapping']).\
            map_elements(lambda row: annotation_fn(row['paragraph_index'], row['info_retrieval_mapping'], row['fact_index']))\
                .alias(f'{model_name}_prompt')
                # alias(f'{model_name}_intersection_label')
    ])
    batch_size = 64
    other_info_gap_prompts = other_info_gap_df[f'{model_name}_prompt'].to_list()
    # iterate over the prompts in batches
    for i in range(0, len(other_info_gap_prompts), batch_size):
        batch = other_info_gap_prompts[i:i+batch_size]
        try:
            other_intersection_labels.extend(predict_local(batch))
        except OutOfMemoryError: # back off to single predictions
            for prompt in batch:
                other_intersection_labels.append(predict_local([prompt]))
        progress.update(len(batch))
    # add the intersection labels to the info gap dataframes
    other_info_gap_df = other_info_gap_df.with_columns([
        pl.lit(other_intersection_labels).alias(f'{model_name}_intersection_label')
    ])

    annotation_fn = partial(annotate_llm, 'en', other_lang_code, en_info_gap_df['person_name'][0], en_info_gap_df, other_info_gap_df)
    en_intersection_labels = []
    en_info_gap_df = en_info_gap_df.with_columns([
        pl.struct(['paragraph_index', 'fact_index',  'info_retrieval_mapping']).\
            map_elements(lambda row: annotation_fn(row['paragraph_index'], row['info_retrieval_mapping'], row['fact_index'])).\
                alias(f'{model_name}_prompt')
    ])
    en_info_gap_prompts = en_info_gap_df[f'{model_name}_prompt'].to_list()
    for i in range(0, len(en_info_gap_prompts), batch_size):
        batch = en_info_gap_prompts[i:i+batch_size]
        try:
            en_intersection_labels.extend(predict_local(batch))
        except OutOfMemoryError:
            for prompt in batch:
                try:
                    en_intersection_labels.append(predict_local([prompt]))
                except OutOfMemoryError:
                    raise ExceptionOOMSingleDataPoint(f"OOM trying to predict a single data point for {en_info_gap_df['person_name'][0]}: {prompt}")
        progress.update(len(batch))
    # add the intersection labels to the info gap dataframes
    en_info_gap_df = en_info_gap_df.with_columns([
        pl.lit(en_intersection_labels).alias(f'{model_name}_intersection_label')
    ])
    return en_info_gap_df, other_info_gap_df, alignment_df

def step_collapse_gpt_labels(gpt_info_gap_dfs, model_intersection_names, **kwargs):
    en_info_gap_df, tgt_info_gap_df, alignment_df = gpt_info_gap_dfs
    def assign_gpt_label(info_gap_frame):
        # NOTE: be careful about what happens here with the 'not intersection' case
        # by accident it has worked out (assigned 'no') but it might not always work out
        return info_gap_frame.with_columns([
            pl.col(f"{model_name}_intersection_label").map_elements(lambda labels: 'yes' if ('y' in [label.strip().lower() for label in labels]) 
                                                        else 'no').alias(f'{model_name}_intersection_label') for model_name in model_intersection_names
        ])
    en_info_gap_df = assign_gpt_label(en_info_gap_df).drop('fact_embed')
    tgt_info_gap_df = assign_gpt_label(tgt_info_gap_df).drop('fact_embed')
    return en_info_gap_df, tgt_info_gap_df, alignment_df