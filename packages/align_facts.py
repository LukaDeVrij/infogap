import ipdb
import numpy as np
from typing import List, Tuple
import polars as pl
from sentence_transformers import SentenceTransformer
import loguru
from packages.constants import HF_CACHE_DIR

logger = loguru.logger

# class Sente
model = SentenceTransformer('sentence-transformers/LaBSE', cache_folder=HF_CACHE_DIR)

def obtain_hubness_measure(fact_embed_df: pl.DataFrame, 
                    background_corpus_sents: List[str]):
    fact_embeds = np.array(fact_embed_df['fact_embed'].to_list())
    background_sent_embeds = model.encode(background_corpus_sents)
    k = 2
    # normalize the background sentence embeddings to unit vectors
    background_sent_embeds = background_sent_embeds / np.linalg.norm(background_sent_embeds, axis=1, keepdims=True)

    # normalize the fact embeddings to unit vectors
    fact_embeds = fact_embeds / np.linalg.norm(fact_embeds, axis=1, keepdims=True)
    # compute the fact-sentence cosine similarity matrix
    fact_sentence_cosine_sim = np.dot(fact_embeds, background_sent_embeds.T)
    top_k_indices = np.argsort(fact_sentence_cosine_sim, axis=1)[:, -k:]
    # using top_k_indices, compute the average cosine similarity between every fact and its nearest neighbour 
    cosine_sim_unnormalized = np.sum(fact_sentence_cosine_sim[np.arange(fact_sentence_cosine_sim.shape[0])[:, None], top_k_indices], axis=1)
    avg_cosine_sim = cosine_sim_unnormalized / (2 * k)
    return fact_embed_df.with_columns([
        pl.lit(avg_cosine_sim).alias('hubness_cosine')
    ])

def pairwise_fact_fact_margin_compute(src_fact_frame: pl.DataFrame, tgt_fact_frame: pl.DataFrame, use_margin_adjustment: bool):
    """Compute the pairwise margin between the source and target fact embeddings.

    Args:
        src_fact_embeds (pl.DataFrame): A numpy array containing the source fact embeddings.
        tgt_fact_embeds (pl.DataFrame): A numpy array containing the target fact embeddings.
        use_margin_adjustment (bool): Whether to use the margin adjustment from Artetxe & Schwenk (2018).
    
    Returns:
        fact_fact_metric (np.array): A numpy array containing the pairwise margin between the source and target fact embeddings.
    """
    src_fact_embeds = np.array(src_fact_frame['fact_embed'].to_list()) 
    tgt_fact_embeds = np.array(tgt_fact_frame['fact_embed'].to_list())
    # normalize the fact embeddings to unit vectors
    src_fact_embeds = src_fact_embeds / np.linalg.norm(src_fact_embeds, axis=1, keepdims=True)
    tgt_fact_embeds = tgt_fact_embeds / np.linalg.norm(tgt_fact_embeds, axis=1, keepdims=True)

    fact_fact_cosine_sim = np.dot(src_fact_embeds, tgt_fact_embeds.T)
    if use_margin_adjustment:
        assert ('hubness_cosine' in src_fact_frame.columns) and ('hubness_cosine' in tgt_fact_frame), logger.error("hubness_cosine column must be present in the dataframes")
        # adjust by subtracting the hubness measure of every english fact along the rows and
        # the hubness measure of every french fact along the columns
        src_hubness_measure = np.array(src_fact_frame['hubness_cosine'])
        tgt_hubness_measure = np.array(tgt_fact_frame['hubness_cosine'])
        fact_fact_margin = fact_fact_cosine_sim - (src_hubness_measure[:, None] + tgt_hubness_measure[None, :])
        fact_fact_metric = fact_fact_margin
    else:
        fact_fact_metric = fact_fact_cosine_sim
    return fact_fact_metric

def pairwise_cosine_similarity(en_fact_embed_df: pl.DataFrame, 
                               fr_fact_embed_df: pl.DataFrame, 
                               use_margin_adjustment, 
                               k=1) -> Tuple[pl.DataFrame, pl.DataFrame]:
    en_fact_embeds = np.array(en_fact_embed_df['fact_embed'].to_list()) 
    fr_fact_embeds = np.array(fr_fact_embed_df['fact_embed'].to_list())
    # normalize the fact embeddings to unit vectors
    en_fact_embeds = en_fact_embeds / np.linalg.norm(en_fact_embeds, axis=1, keepdims=True)
    fr_fact_embeds = fr_fact_embeds / np.linalg.norm(fr_fact_embeds, axis=1, keepdims=True)

    fact_fact_cosine_sim = np.dot(en_fact_embeds, fr_fact_embeds.T)
    if use_margin_adjustment:
        # adjust by subtracting the hubness measure of every english fact along the rows and
        # the hubness measure of every french fact along the columns
        en_hubness_measure = np.array(en_fact_embed_df['hubness_cosine'])
        fr_hubness_measure = np.array(fr_fact_embed_df['hubness_cosine'])
        fact_fact_margin = fact_fact_cosine_sim - (en_hubness_measure[:, None] + fr_hubness_measure[None, :])
        fact_fact_metric = fact_fact_margin
    else:
        fact_fact_metric = fact_fact_cosine_sim
    
    # compute en -> fr alignment.
    # add a column that is the indices of the top 3 aligned facts from en facts to fr ones.
    # also put a column that is a tuple of size 3, containing the margin between the en fact and each of the top 3 fr facts
    # the margin is the difference between the cosine similarity of the en fact with the fr fact along with the sum of the hubness measures of the en fact and the fr fact

    top_k_indices = np.argsort(fact_fact_metric, axis=1)[:, -k:]
    top_k_metric = np.sort(fact_fact_metric, axis=1)[:, -k:]
    
    result_frame = en_fact_embed_df.with_columns([
        pl.lit(top_k_indices).alias('top_k_indices'),
        pl.lit(top_k_metric).alias(('top_k_margin' if use_margin_adjustment else 'top_k_cosine_sim'))
    ])
    # add the sentences for the top k indices from the fr fact dataframe
    top_k_fr_sentences = []
    for i in range(k):
        top_k_fr_sentences.append(fr_fact_embed_df['fact'][top_k_indices[:, i]])
    # it should be a single column with a list of size 3
    # right now, it is 3 columns with a single element in each column
    # here's the fix
    top_k_fr_sentences = np.array(top_k_fr_sentences).T.tolist()
    result_frame = result_frame.with_columns([
        pl.lit(top_k_fr_sentences).alias('top_k_fr_sentences')
    ])
    return result_frame

def align_facts(src_lang_fact_df: pl.DataFrame, 
                tgt_lang_fact_df: pl.DataFrame, 
                en_sentences: List[str], fr_sentences: List[str]):
    """
    """
    src_fact_embed_df = obtain_hubness_measure(src_lang_fact_df, en_sentences)
    tgt_fact_embed_df = obtain_hubness_measure(tgt_lang_fact_df, fr_sentences)

    # compute the pairwise cosine similarity between the en and fr fact embeddings
    margin_matrix = pairwise_fact_fact_margin_compute(src_fact_embed_df, tgt_fact_embed_df, use_margin_adjustment=True)
    # # compute the pairwise cosine similarity between the en and fr fact embeddings

    # target_columns = ['fact_paragraph_num', 'fact', 'top_k_margin', 'top_k_fr_sentences']
    # # sort by top_k_margin in descending order, then group by fact and take first 3 rows
    # src_result_frame = src_result_frame.select(target_columns)\
    #                 .explode(['top_k_fr_sentences', 'top_k_margin'])\
    #                 .sort('top_k_margin', descending=True).group_by('fact').head(3)
    return margin_matrix
