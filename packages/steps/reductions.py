from typing import List, Tuple
import polars as pl

# TODO: have to change this
def reduce_info_gaps(en_fr_info_gaps: List[Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]]) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    # concatenate the en dataframes and the fr dataframes

    en_info_gaps = pl.concat([(info_gap[0].drop('flan-large_prompt') if 'flan-large_prompt' in info_gap[0].columns else info_gap[0]) for info_gap in en_fr_info_gaps])
    fr_info_gaps = pl.concat([(info_gap[1].drop('flan-large_prompt') if 'flan-large_prompt' in info_gap[1].columns else info_gap[1]) for info_gap in en_fr_info_gaps])
    alignment_dfs = pl.concat([pl.from_pandas(info_gap[2]) for info_gap in en_fr_info_gaps])
    # en_info_gaps = pl.concat(en_info_gaps)
    # fr_info_gaps = pl.concat(fr_info_gaps)
    # alignment_dfs = pl.concat(alignment_dfs)
    # en_info_gaps = assign_gpt_label(en_info_gaps)
    # fr_info_gaps = assign_gpt_label(fr_info_gaps)
    
    return en_info_gaps, fr_info_gaps, alignment_dfs

def reduce_paragraph_ablation(abln_info_gap_dfs: List[pl.DataFrame]):
    """
    Columns: 
        - src_contexts: List[str]
        - tgt_contexts: List[List[str]]
        - gpt-4_intersection_label: str
        - language: str
        - person_name: str
    """
    all_people_info_gap_frame = pl.concat(abln_info_gap_dfs)
    # rename the 'gpt-4_intersection_label' column to 'gpt-4_paragraph_abln_label'
    all_people_info_gap_frame = all_people_info_gap_frame.with_columns([
        pl.col('gpt-4_intersection_label').alias('gpt-4_paragraph_abln_label')
    ])
    return all_people_info_gap_frame

# TODO: have to change this
def reduce_caa_classifications(en_fr_connotations: List[Tuple[pl.DataFrame, pl.DataFrame]]) -> pl.DataFrame:
    # assert that all the frames are of the same length
    en_connotations = pl.concat([connotation[0] for connotation in en_fr_connotations]).with_columns(pl.lit('en').alias('src_lang_code'))
    fr_connotations = pl.concat([connotation[1] for connotation in en_fr_connotations]).with_columns(pl.lit('fr').alias('src_lang_code'))
    return pl.concat([en_connotations, fr_connotations])