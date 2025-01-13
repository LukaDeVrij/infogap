import ipdb
from typing import List, Union
from wikipedia_edit_scrape_tool import Paragraph, Header
from sklearn.metrics.pairwise import cosine_similarity
import polars as pl
from typing import List
from tqdm import tqdm
from collections import OrderedDict
from flowmason import SingletonStep, MapReduceStep
from .constants import HF_CACHE_DIR
from sentence_transformers import SentenceTransformer
from nltk import sent_tokenize
import numpy as np

def forced_align_fact_to_paragraph(fact_df: pl.DataFrame, paragraph: Paragraph, 
                                   hubness_measure: np.array,
                                   model: SentenceTransformer):
    # return a list of indices of length {fact_df} that map to indices in paragraph_df. 
    fact_sents = fact_df["fact"].to_list()
    fact_embeds = model.encode(fact_sents)
    paragraph_sents = sent_tokenize(paragraph.clean_text)
    assert hubness_measure.shape[0] == len(paragraph_sents)
    full_sent_embeds = model.encode(paragraph_sents)
    # NOTE: if we go back to forced alignfrment, need to ensure that the facts have not been shuffled
    sim_matrix = cosine_similarity(full_sent_embeds, fact_embeds) # shape: (len(paragraph_sents), len(fact_sents))
    sim_matrix = sim_matrix - hubness_measure[:, None] # adjust the similarity matrix by subtracting the hubness measure, accounting for sentences that are very hubby (similar to many facts/sentences)
    assert sim_matrix.shape[0] == len(paragraph_sents) and sim_matrix.shape[1] == len(fact_sents)
    alignment_inds = sim_matrix.argmax(axis=0) # shape: (len(fact_sents))
    return [paragraph_sents[alignment_ind] for alignment_ind in alignment_inds]

def forced_align(facts: List[str], sentences: str, 
                 hubness_measure: np.array,
                 model):
    # the list of facts is longer than the list of sentences
    # the conditions of first alignment are:
        # the first fact must align with the first sentence
        # the last fact must align with the last sentence
        # the alignment is monotonic
    sentences = sent_tokenize(sentences)
    opt_dist = np.ones((len(facts), len(sentences) )) * np.inf
    fact_embeds = model.encode(facts)
    sentence_embeds = model.encode(sentences)
    fact_embeds = fact_embeds / np.linalg.norm(fact_embeds, axis=1, keepdims=True)
    sentence_embeds = sentence_embeds / np.linalg.norm(sentence_embeds, axis=1, keepdims=True)

    sim_matrix = cosine_similarity(sentence_embeds, fact_embeds) # shape: (len(paragraph_sents), len(fact_sents))
    sim_matrix = sim_matrix - hubness_measure[:, None] # adjust the similarity matrix by subtracting the hubness measure, accounting for sentences that are very hubby (similar to many facts/sentences)

    i = 0
    j = 0

    num_facts = len(facts)
    num_sents = len(sentences)
    opt_dist[0, 0] = np.linalg.norm(fact_embeds[0] - sentence_embeds[0])
    # at any given time, the distance between i and j can be at most num_facts - num_sents
    for i in range(1, num_facts):
        for j in range(0, num_sents):
            dist = -sim_matrix[j, i]
            opt_dist[i, j] = dist + min( # adding the dist encodes the fact that the last two lements are aligned.
                opt_dist[i-1, j-1],
                opt_dist[i-1, j] # the cost of aligning the fact with the previous sentence
            )
    # now we need to backtrack to get the alignment
    reconstructed = [sentences[-1]] # the last sentence is always aligned with the last fact
    if len(facts) == 1:
        return reconstructed
    i = num_facts - 1 # last fact has been taken care of
    j = num_sents - 1
    while i > 0 and j > 0:
        if opt_dist[i-1, j-1] < opt_dist[i-1, j]:
            i -= 1
            j -= 1
            reconstructed.insert(0, sentences[j])
        else:
            i -= 1
            reconstructed.insert(0, sentences[j])
    # insert the first sentence for as many of the remaining facts as possible
    for _ in range(i):
        reconstructed.insert(0, sentences[0]) # the first sentence is always aligned with the first fact
    assert len(reconstructed) == len(facts), ipdb.set_trace()
    return reconstructed

def compute_hubness(paragraph: Paragraph, other_fact_blocks: pl.DataFrame, num_hubness_compute, model):
    sents = sent_tokenize(paragraph.clean_text)
    other_facts = other_fact_blocks['fact'].to_list()
    if other_facts== []: # if there is only one paragraph?
        return np.zeros(len(sents))
    # compute the cosine similarity between the sentences in the paragraph and the other paragraphs
    sent_embeds = model.encode(sents)
    other_sent_embeds = model.encode(other_facts)
    sent_embeds = sent_embeds / np.linalg.norm(sent_embeds, axis=1, keepdims=True)
    other_sent_embeds = other_sent_embeds / np.linalg.norm(other_sent_embeds, axis=1, keepdims=True)
    cosine_sim = np.dot(sent_embeds, other_sent_embeds.T)
    top_k_indices = np.argsort(cosine_sim, axis=1)[:, -num_hubness_compute:]
    cosine_sim_unnormalized = np.sum(cosine_sim[np.arange(cosine_sim.shape[0])[:, None], top_k_indices], axis=1)
    avg_cosine_sim = cosine_sim_unnormalized / num_hubness_compute # compute the hubness measure
    return avg_cosine_sim

def step_forced_align_facts_to_paragraph(en_fr_info_gaps, en_content_blocks: List[Union[Paragraph, Header]], 
                                        fr_content_blocks: List[Union[Paragraph, Header]], 
                                        pronoun: str, **kwargs):
    # set en_paragraphs to all the elements of type Paragraph in en_content_blocks
    en_paragraphs = [block for block in en_content_blocks if isinstance(block, Paragraph)]
    fr_paragraphs = [block for block in fr_content_blocks if isinstance(block, Paragraph)]
    model = SentenceTransformer('sentence-transformers/LaBSE', cache_folder=HF_CACHE_DIR)
    en_info_gaps, fr_info_gaps, alignment_dfs = [info_gap for info_gap in en_fr_info_gaps]

    # NOTE: this function assumes that only a single bio is being processed at a time.
    def _align_facts_to_gt_sentences(info_gap_df, paragraphs):
        algn_aug_info_gaps = []
        for paragraph_index in info_gap_df['paragraph_index'].unique():
            other_fact_blocks = info_gap_df.filter(pl.col('paragraph_index') != paragraph_index)
            paragraph_hubnesses = compute_hubness(paragraphs[paragraph_index], other_fact_blocks, 5, model) 
            paragraph_df = info_gap_df.filter(pl.col('paragraph_index') == paragraph_index)
            # aligned_sentences = forced_align_fact_to_paragraph(paragraph_df, paragraphs[paragraph_index], paragraph_hubnesses, model)

            # TODO: need to do the hubness adjustment still.
            aligned_sentences = forced_align(paragraph_df['fact'].to_list(), paragraphs[paragraph_index].clean_text, 
                                             paragraph_hubnesses,
                                             model)
            # add the aligned sentences to the paragraph_df
            paragraph_df = paragraph_df.with_columns(pl.Series(name='aligned_sentence', values=aligned_sentences))
            algn_aug_info_gaps.append(paragraph_df)
        result_info_gap_df = pl.concat(algn_aug_info_gaps)
        return result_info_gap_df
    en_info_gaps = _align_facts_to_gt_sentences(en_info_gaps, en_paragraphs)
    fr_info_gaps = _align_facts_to_gt_sentences(fr_info_gaps, fr_paragraphs)
    # add the pronoun to the info_gap_df
    en_info_gaps = en_info_gaps.with_columns(pl.lit(pronoun).alias('pronoun'))
    fr_info_gaps = fr_info_gaps.with_columns(pl.lit(pronoun).alias('pronoun'))
    return en_info_gaps, fr_info_gaps, alignment_dfs    
    # eventually, we will want to return the alignment information into the information gap dataframe.

def step_forced_align_en_tgt_facts_to_paragraph(en_tgt_info_gaps, en_content_blocks: List[Union[Paragraph, Header]], 
                                        tgt_content_blocks: List[Union[Paragraph, Header]], 
                                        pronoun: str, **kwargs):
    # set en_paragraphs to all the elements of type Paragraph in en_content_blocks
    en_paragraphs = [block for block in en_content_blocks if isinstance(block, Paragraph)]
    tgt_paragraphs = [block for block in tgt_content_blocks if isinstance(block, Paragraph)]
    model = SentenceTransformer('sentence-transformers/LaBSE', cache_folder=HF_CACHE_DIR)
    en_info_gaps, tgt_info_gaps, alignment_dfs = [info_gap for info_gap in en_tgt_info_gaps]

    # NOTE: this function assumes that only a single bio is being processed at a time.
    progress = tqdm(total=len(en_info_gaps['paragraph_index'].unique()) + len(tgt_info_gaps['paragraph_index'].unique()))
    def _align_facts_to_gt_sentences(info_gap_df, paragraphs):
        algn_aug_info_gaps = []
        for paragraph_index in info_gap_df['paragraph_index'].unique():
            other_fact_blocks = info_gap_df.filter(pl.col('paragraph_index') != paragraph_index)
            paragraph_hubnesses = compute_hubness(paragraphs[paragraph_index], other_fact_blocks, 5, model) 
            paragraph_df = info_gap_df.filter(pl.col('paragraph_index') == paragraph_index)
            # aligned_sentences = forced_align_fact_to_paragraph(paragraph_df, paragraphs[paragraph_index], paragraph_hubnesses, model)

            # TODO: need to do the hubness adjustment still.
            aligned_sentences = forced_align(paragraph_df['fact'].to_list(), paragraphs[paragraph_index].clean_text, 
                                             paragraph_hubnesses,
                                             model)
            # add the aligned sentences to the paragraph_df
            paragraph_df = paragraph_df.with_columns(pl.Series(name='aligned_sentence', values=aligned_sentences))
            algn_aug_info_gaps.append(paragraph_df)
            progress.update(1)
        result_info_gap_df = pl.concat(algn_aug_info_gaps)
        return result_info_gap_df
    en_info_gaps = _align_facts_to_gt_sentences(en_info_gaps, en_paragraphs)
    tgt_info_gaps = _align_facts_to_gt_sentences(tgt_info_gaps, tgt_paragraphs)
    # add the pronoun to the info_gap_df
    en_info_gaps = en_info_gaps.with_columns(pl.lit(pronoun).alias('pronoun'))
    tgt_info_gaps = tgt_info_gaps.with_columns(pl.lit(pronoun).alias('pronoun'))
    return en_info_gaps, tgt_info_gaps, alignment_dfs    

def construct_info_diff_caa_step_dict():
    info_diff_caa_dict = OrderedDict()
    info_diff_caa_dict['step_forced_align_fact_to_sent'] = SingletonStep(step_forced_align_facts_to_paragraph, {
        'en_fr_info_gaps': 'step_refine_intersection_label', 
        'en_paragraphs': 'step_load_degeneres_bio',
        'fr_paragraphs': 'step_load_degeneres_bio_fr'
    })

