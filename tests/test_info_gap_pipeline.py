import time
from openai import OpenAI
import polars as pl
from nltk import sent_tokenize
import pytest
from functools import partial
import ipdb
from flowmason import conduct, SingletonStep, load_artifact, load_artifact, load_artifact_with_step_name
from packages.steps.map_dicts import get_en_fr_info_diff_map_dict
from packages.steps.info_diff_steps import step_retrieve_en_content_blocks, step_generate_facts, step_compute_info_gap_reasoning, step_retrieve_fr_content_blocks
from packages.gpt_query import ask_gpt_about_fact_intersection
from wikipedia_edit_scrape_tool import get_text, Header, Paragraph 
from dotenv import dotenv_values
import loguru

logger = loguru.logger

@pytest.fixture
def metadata_gabriel_attal():
    bio_id = "Gabriel_Attal"
    step_dict = get_en_fr_info_diff_map_dict(bio_id, bio_id, "Gabriel Attal")
    metadata = conduct("test_cache", step_dict, "info_diff_implementation_test_logs")
    yield metadata

@pytest.fixture
def directional_paragraph_alignments(metadata_gabriel_attal):
    # TODO: retrieve the directional paragraph alignments from the cache using load_artifact.
    directional_paragraph_alignments = load_artifact_with_step_name(metadata_gabriel_attal, "step_align_fact_paragraphs")
    yield directional_paragraph_alignments

@pytest.fixture
def get_en_facts(metadata):
    # TODO: retrieve the English content blocks from the cache using load_artifact.
    en_fact_blocks = load_artifact(metadata, "step_generate_facts")
    yield en_fact_blocks


@pytest.fixture
def fr_content_blocks(metadata_gabriel_attal):
    # TODO: retrieve the French content blocks from the cache using load_artifact.
    fr_fact_blocks = load_artifact_with_step_name(metadata_gabriel_attal, 'step_get_fr_content_blocks')
    ipdb.set_trace()
    yield fr_fact_blocks

@pytest.fixture
def en_content_blocks(metadata_gabriel_attal):
    en_content_blocks = load_artifact_with_step_name(metadata_gabriel_attal, 'step_get_en_content_blocks')
    ipdb.set_trace()
    yield en_content_blocks

@pytest.fixture
def en_facts(metadata_gabriel_attal):
    en_facts = load_artifact_with_step_name(metadata_gabriel_attal, 'step_generate_facts')
    yield en_facts

@pytest.fixture
def fr_facts(metadata_gabriel_attal):
    fr_facts = load_artifact_with_step_name(metadata_gabriel_attal, 'step_generate_facts_fr')
    yield fr_facts

@pytest.fixture
def union_alignment_df(metadata_gabriel_attal):
    union_alignment_df = load_artifact_with_step_name(metadata_gabriel_attal, 'step_union_fact_paragraphs')
    yield union_alignment_df

@pytest.fixture
def retrieval_candidate_frames(metadata_gabriel_attal):
    retrieval_candidate_frame = load_artifact_with_step_name(metadata_gabriel_attal, 'step_find_retrieval_candidates')
    yield retrieval_candidate_frame

@pytest.fixture
def gpt_label_frames(metadata_gabriel_attal):
    gpt_label_frames = load_artifact_with_step_name(metadata_gabriel_attal, 'step_reasoning_intersection_label')
    yield gpt_label_frames

@pytest.fixture
def gpt_collapsed_labels(metadata_gabriel_attal):
    gpt_collapsed_labels = load_artifact_with_step_name(metadata_gabriel_attal, 'step_collapse_gpt_labels')
    yield gpt_collapsed_labels

@pytest.fixture
def info_gap_with_aligned_sentences(metadata_gabriel_attal):
    info_gap_with_aligned_sentences = load_artifact_with_step_name(metadata_gabriel_attal, 'step_add_fact_to_sent_alignment_info')
    yield info_gap_with_aligned_sentences

def test_get_fr_content_blocks(fr_content_blocks):
    # voir_aussi_header = next(filter(lambda x: isinstance(x, Header) and x.text == "Voir aussi" , fr_fact_blocks))
    # assert that the voir aussi header block does not exist in fact_fr_blocks
    assert not any(isinstance(x, Header) and x.text == "Voir aussi" for x in fr_content_blocks)

def test_directional_alignments(en_facts, fr_facts, directional_paragraph_alignments):
    # assert that the shape of directional_paragraph_alignments[0] == (len(en_facts), len(fr_facts))
    # assert that the shape of directional_paragraph_alignments[1] == (len(fr_facts), len(en_facts))
    assert directional_paragraph_alignments[0].shape == (len(en_facts), len(fr_facts))
    assert directional_paragraph_alignments[1].shape == (len(fr_facts), len(en_facts))


def test_step_union_alignments(union_alignment_df, 
                               en_content_blocks, 
                               fr_content_blocks):
    # assert that the number of unique values in union_alignment_df['en_paragraph_index'] == len(en_content_blocks)
    # assert that the number of unique values in union_alignment_df['fr_paragraph_index'] == len(fr_content_blocks)
    assert len(union_alignment_df['en_paragraph_index'].unique()) == len(list(filter(lambda x: isinstance(x, Paragraph), en_content_blocks)))
    assert len(union_alignment_df['fr_paragraph_index'].unique()) == len(list(filter(lambda x: isinstance(x, Paragraph), fr_content_blocks)))
    pass

def test_step_retrieve_aligned_fact_candidates(retrieval_candidate_frames, en_facts, fr_facts):
    en_info_gap_df, fr_info_gap_df, _ = retrieval_candidate_frames
    assert len(en_info_gap_df) == sum([len(fact_block) for fact_block in en_facts])
    assert len(fr_info_gap_df) == sum([len(fact_block) for fact_block in fr_facts])


def test_step_test_alignment_verification(gpt_label_frames, retrieval_candidate_frames):
    en_gpt_label_frame = gpt_label_frames[0]
    en_info_gap = retrieval_candidate_frames[0]
    assert len(en_gpt_label_frame) == len(en_info_gap)

def test_step_collapse_gpt_predictions(gpt_collapsed_labels):
    ipdb.set_trace()
    assert set(gpt_collapsed_labels[0]['final_intersection_label'].unique()) == set(['yes', 'no'])
    assert isinstance(gpt_collapsed_labels[0], pl.DataFrame)
    # assert 'fact_embed' is not in the columns
    assert 'fact_embed' not in gpt_collapsed_labels[0].columns

# TODO: this test suggests that we should fix the forced alignment step. Can look into this later.
def test_step_forced_align_facts_to_paragraph(info_gap_with_aligned_sentences, en_content_blocks, fr_content_blocks):
    en_info_gap_df, fr_info_gap_df, _ = info_gap_with_aligned_sentences
    num_sents_en = sum([len(sent_tokenize(paragraph.clean_text)) for paragraph in en_content_blocks if isinstance(paragraph, Paragraph)])
    num_sents_fr = sum([len(sent_tokenize(paragraph.clean_text)) for paragraph in fr_content_blocks if isinstance(paragraph, Paragraph)])
    fr_paragraphs = [block for block in fr_content_blocks if isinstance(block, Paragraph)]
    # create a dataframe with two columns: the paragraph index and a sentence that the pargraph contains. There should
    # be a row for each sentence in each paragraph.

    fr_paragraph_frames = []
    for i in range(len(fr_paragraphs)):
        paragraph = fr_paragraphs[i]
        sentences = sent_tokenize(paragraph.clean_text)
        paragraph_index = [i] * len(sentences)
        fr_paragraph_frames.append(pl.DataFrame({'paragraph_index': paragraph_index, 'sentence': sentences}))
    fr_paragraph_frame = pl.concat(fr_paragraph_frames)
    ipdb.set_trace()
    assert 'aligned_sentence' in en_info_gap_df.columns
    assert 'aligned_sentence' in fr_info_gap_df.columns
    assert num_sents_en - len(en_info_gap_df['aligned_sentence'].unique()) < 50
    assert num_sents_fr - len(fr_info_gap_df['aligned_sentence'].unique()) < 50
    if num_sents_fr - len(fr_info_gap_df['aligned_sentence'].unique()) > 10:
        logger.warning(f"Number of sentences in French: {num_sents_fr} but number of unique aligned sentences: {len(fr_info_gap_df['aligned_sentence'].unique())}")
    ipdb.set_trace()

def test_fact_caching():
    en_blocks = step_retrieve_en_content_blocks("Gabriel_Attal") 
    start_time = time.time()
    en_facts = step_generate_facts(en_blocks, "en", "Gabriel Attal")
    end_time = time.time()
    first_len = len(en_facts)

    first_run_time = end_time - start_time
    start_time = time.time()
    en_facts = step_generate_facts(en_blocks, "en", "Gabriel Attal")
    end_time = time.time()
    second_run_time = end_time - start_time
    second_len = len(en_facts)
    assert first_run_time > second_run_time
    assert first_len == second_len

def test_fact_generation_ellen_fr():
    fr_blocks = step_retrieve_fr_content_blocks("Ellen_DeGeneres")
    fr_facts = step_generate_facts(fr_blocks, "fr", "Ellen DeGeneres")
    ipdb.set_trace()

def test_fact_intersection_caching(retrieval_candidate_frames): 
    start_time = time.time()
    step_compute_info_gap_reasoning(retrieval_candidate_frames, "Gabriel Attal")
    end_time = time.time()
    first_run_time = end_time - start_time

    start_time = time.time()
    step_compute_info_gap_reasoning(retrieval_candidate_frames, "Gabriel Attal")
    end_time = time.time()
    second_run_time = end_time - start_time
    assert first_run_time > second_run_time

def test_fact_intersection_query():
    config = dotenv_values(".env")
    key = config["THE_KEY"] 
    ask_about_intersection = partial(ask_gpt_about_fact_intersection, OpenAI(api_key=key))
    cache = {}
    src_lang = 'fr'
    tgt_context = [
        ['Attal was born on 16 March 1989 in Clamart, Île-de-France.', 'He grew up in the 13th and 14th arrondissements of Paris with three sisters.', 'His father, Yves Attal, was a lawyer and film producer.'],
        ['Attal attended École alsacienne, an exclusive private school in the 6th arrondissement of Paris.']
    ]
    src_context = ['Gabriel Attal est né le 16 mars 1989.']
    person_name = "Gabriel Attal"
    input_prompt, response_content, total_tokens = ask_about_intersection(cache, src_lang, src_context, tgt_context, person_name)
    assert len(eval(response_content)) == len(tgt_context)
    return input_prompt, response_content, total_tokens

def test_content_block_retrieval_abdellah():
    fr_content_blocks = step_retrieve_fr_content_blocks("Abdellah Taïa")
    # assert that there is no paragraph containing the text "le livre commence comme un monologue"
    paragraphs = [block for block in fr_content_blocks if isinstance(block, Paragraph)]
    assert not any(["Le livre commence comme un monologue" in paragraph.clean_text for paragraph in paragraphs])