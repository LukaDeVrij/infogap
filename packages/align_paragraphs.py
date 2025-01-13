from scipy.special import softmax
import ipdb
from typing import List, Tuple, Optional, Callable
from scipy.sparse import coo_matrix, csc_matrix
import loguru
from functools import partial
import polars as pl
import numpy as np
# import dataclass
from dataclasses import dataclass
# from wikipedia_edit_scrape_tool import Paragraph 
# https://huggingface.co/sentence-transformers/LaBSE

from .gpt_query import FactParagraph
from .align_facts import pairwise_fact_fact_margin_compute, obtain_hubness_measure

logger = loguru.logger

def compute_directional_paragraph_alignment(src_lang_fact_df: pl.DataFrame, tgt_lang_fact_df: pl.DataFrame, 
                                            pairwise_fact_comparison_fn: Optional[Callable] = None) -> Tuple[np.array, np.array]:
    """Compute the alignment matrix between the source and target language paragraphs.
    It is directional (i.e., which paragraphs in the source language align to which paragraphs in the target language).

    Args:
        src_lang_fact_df (pl.DataFrame): A dataframe containing the source language facts and the paragraph index
        tgt_lang_fact_df (pl.DataFrame): A dataframe containing the target language facts and the paragraph index
        pairwise_fact_comparison_fn (Optional[Callable]): A function that takes in two dataframes containing facts and returns a 
            matrix of shape (num_facts_lang_one, num_facts_lang_two) containing the pairwise association strength between the facts.
    
    Returns:
        src_tgt_alignment (np.array): A numpy array containing the alignment matrix between the source and target language paragraphs.
            The alignment matrix is represented as a vector of length num_paragraphs_src, where the 
            value at index i is the index of the paragraph in lang_two that paragraph i in lang_one aligns to.
        mapping_averages (np.array): A numpy array containing the average cosine similarity 
            between the source and target language paragraphs. It is a matrix of shape (num_paragraphs_lang_one, num_paragraphs_lang_two)
    """
    pairwise_margin_matrix = pairwise_fact_comparison_fn(src_lang_fact_df, 
                                                               tgt_lang_fact_df) # numpy array
    assert pairwise_margin_matrix.shape[0] == len(src_lang_fact_df)
    assert pairwise_margin_matrix.shape[1] == len(tgt_lang_fact_df)

    tgt_paragraph_index_counts = tgt_lang_fact_df['paragraph_index'].sort(descending=False).value_counts()
    src_paragraph_index_counts = src_lang_fact_df['paragraph_index'].sort(descending=False).value_counts()
    tgt_paragraph_start_pointer = 0
    src_tgt_paragraph_associations = []
    for tgt_paragraph_row_size in tgt_paragraph_index_counts.iter_rows(named=True):
        num_sents_in_tgt_paragraph = tgt_paragraph_row_size['counts']
        # mapping_averages.append(np.mean(pairwise_margin_matrix[:, curr_paragraph_index:curr_paragraph_index + count]))
        # slice out columns of the pairwise_margin_matrix that correspond to the current paragraph 
        # and compute the average across the columns
        tgt_paragraph_assoc_strength = np.max(pairwise_margin_matrix[:, tgt_paragraph_start_pointer:tgt_paragraph_start_pointer + num_sents_in_tgt_paragraph], axis=1) # num source sentences x num target paragraphs 
        assert (tgt_paragraph_assoc_strength.shape[0] == len(src_lang_fact_df)) 
        tgt_paragraph_start_pointer += num_sents_in_tgt_paragraph
        # use median to collapse to num source paragraphs x num target paragraphs
        # create an empty list for storing the median association strength between the source paragraph and this target paragraph

        src_2_tgt_paragraph_assoc_strength = [] 
        src_paragraph_start_pointer = 0
        for src_paragraph_row_size in src_paragraph_index_counts.iter_rows(named=True):
            num_sents_in_src_paragraph = src_paragraph_row_size['counts']
            src_2_tgt_paragraph_assoc_strength.append(np.median(tgt_paragraph_assoc_strength[src_paragraph_start_pointer:src_paragraph_start_pointer + num_sents_in_src_paragraph]))
            src_paragraph_start_pointer += num_sents_in_src_paragraph
        src_tgt_paragraph_associations.append(np.array(src_2_tgt_paragraph_assoc_strength))
    return np.array(src_tgt_paragraph_associations).T

def compute_algn_strngths(lang_one_facts_df: pl.DataFrame, lang_two_facts_df: pl.DataFrame, 
                          comparison_fn_name: str, **kwargs) -> Tuple[np.array, np.array]:
    """
    Aligns paragraphs from the English and French Wikipedia articles.

    Args:
        lang_one_facts_df (pl.DataFrame): A dataframe containing the English facts and the paragraph index (i.e., the 
            paragraph index in the English Wikipedia article)
        lang_two_facts_df (pl.DataFrame): A dataframe containing the French facts and the paragraph index.
        comparison_fn_name (str): The name of the function to use for comparing the facts. 
    """
    # sort the dataframes by the paragraph index
    lang_one_facts_df = lang_one_facts_df.sort('paragraph_index', descending=False)
    lang_two_facts_df = lang_two_facts_df.sort('paragraph_index', descending=False)
    if comparison_fn_name == 'hubness_margin':
        assert len(lang_one_facts_df.columns) == 3, logger.error("en_facts_df parameter must have exactly 3 columns")

        lang_one_background_sentences = lang_one_facts_df['fact'].to_list()
        lang_two_background_sentences = lang_two_facts_df['fact'].to_list()

        lang_one_facts_df = obtain_hubness_measure(lang_one_facts_df, lang_one_background_sentences) # add hubness_cosine column
        lang_two_facts_df = obtain_hubness_measure(lang_two_facts_df, lang_two_background_sentences)
        comparison_fn = partial(pairwise_fact_fact_margin_compute, use_margin_adjustment=True)
    elif comparison_fn_name == 'bert_score':
        def _compare_facts_bertscore_df(lang_one_facts_df: pl.DataFrame, lang_two_facts_df: pl.DataFrame) -> np.array:
            lang_one_facts = lang_one_facts_df['fact'].to_list()
            lang_two_facts = lang_two_facts_df['fact'].to_list()
            return compute_all_pairs_bertscore(lang_one_facts, lang_two_facts)
        comparison_fn = _compare_facts_bertscore_df

    # compute the alignment matrix for the lang_one -> lang_two direction
    lang_one_to_lang_two_mapping_averages = compute_directional_paragraph_alignment(
        lang_one_facts_df, lang_two_facts_df, comparison_fn)
    # assert that the shape of lang_one_to_lang_two_alignment is (num_paragraphs_lang_one, ) 
    assert lang_one_to_lang_two_mapping_averages.shape[0] == len(lang_one_facts_df['paragraph_index'].unique()) \
        and lang_one_to_lang_two_mapping_averages.shape[1] == len(lang_two_facts_df['paragraph_index'].unique())
    # compute the alignment matrix for the lang_two -> lang_one direction
    lang_two_to_lang_one_mapping_averages = compute_directional_paragraph_alignment(lang_two_facts_df, lang_one_facts_df, comparison_fn)
    assert lang_two_to_lang_one_mapping_averages.shape[0] == len(lang_two_facts_df['paragraph_index'].unique()) \
        and lang_two_to_lang_one_mapping_averages.shape[1] == len(lang_one_facts_df['paragraph_index'].unique())

    return lang_one_to_lang_two_mapping_averages, lang_two_to_lang_one_mapping_averages

def prune_directional_alignments(para_para_assoc_strength: np.array) -> np.array:
    """Prune the directional paragraph alignment matrix.

    Args:
        para_para_assoc_strength (np.array): A numpy array containing the pairwise association strength between the source and target paragraphs.
            The entries are scalars. The rows are the source paragraphs and the columns are the target paragraphs.

    Returns: 
        np.array
    """
    # calling argmax on every row of the alignment matrix will give us the index of the target paragraph that the source paragraph aligns to
    # however, some source paragraphs should be isolated (i.e., they should not align to any target paragraphs)

    # to find the paragraphs that should be isolated, take the max value of the row.
    # then do a bootstrap percentile test to see if the max value is significantly greater than the other values in the row
    # the p value should be 1 / {para_para_assoc_strength.shape[1] - 1} (i.e., the number of target paragraphs - 1).
    # use B = 1000 bootstrap samples

    pruned_alignment_mat = np.zeros_like(para_para_assoc_strength)
    for i in range(para_para_assoc_strength.shape[0]):
        # get the max value of the row
        max_value = np.max(para_para_assoc_strength[i])
        # flattened_para_para_assoc_strength = para_para_assoc_strength[]
        # get the array excluding the current row i
        flattened_para_para_assoc_strength = np.delete(para_para_assoc_strength, i, axis=0).flatten()
        # compute the empirical bootstrap p value


        bootstrap_p_value = (flattened_para_para_assoc_strength > max_value).sum() / len(flattened_para_para_assoc_strength)
        # if the p value is less than 0.05, then we should not isolate this paragraph
        if bootstrap_p_value < (1 / (para_para_assoc_strength.shape[1])):
            pruned_alignment_mat[i, np.argmax(para_para_assoc_strength[i])] = 1
    # create a sparse CSC matrix with the entries
    return pruned_alignment_mat

def union_forced_alignment(dir_algn_mat_1: np.array, dir_algn_mat_2: np.array) -> np.array:
    """Take the union of the paragraph alignment matrices.

    Args:
        dir_algn_mat_1 (np.array): A numpy array containing one directional alignment matrix. (n)
        dir_algn_mat_2 (np.array): A numpy array containing the other directional alignment matrix. (m)
    
    Returns:
        np.array: A numpy array containing the union of the two directional alignment matrices. (n x m)
    """
    union_alignment = np.zeros((len(dir_algn_mat_1), len(dir_algn_mat_2)))
    for i in range(len(dir_algn_mat_1)):
        for j in range(len(dir_algn_mat_2)):
            if dir_algn_mat_1[i] == j and dir_algn_mat_2[j] == i:
                union_alignment[i, j] = 2
            elif dir_algn_mat_1[i] == j or dir_algn_mat_2[j] == i:
                union_alignment[i, j] = 1
    return union_alignment

def union_pruned_alignment(dir_algn_mat_1: np.array, dir_algn_mat_2: np.array) -> np.array:
    """Take the union of the paragraph alignment matrices.

    Args:
        dir_algn_mat_1 (np.array): A numpy array containing one directional alignment matrix. (n x m)
        dir_algn_mat_2 (np.array): A numpy array containing the other directional alignment matrix. (m x n) 
    """
    return dir_algn_mat_1 + dir_algn_mat_2.T