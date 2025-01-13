import pytest
from flowmason import conduct, load_artifact, load_artifact_with_step_name
import ipdb
from wikipedia_edit_scrape_tool import get_text, Header, Paragraph 

from packages.steps.map_dicts import get_en_fr_info_diff_map_dict_flan
from packages.info_diff_caa import step_forced_align_facts_to_paragraph

@pytest.fixture
def metadata_perfume_genius():
    en_bio_id = "Perfume_Genius"
    fr_bio_id = "Perfume_Genius"
    step_dict = get_en_fr_info_diff_map_dict_flan(en_bio_id, fr_bio_id, person_name="Perfume Genius")
    metadata = conduct("test_cache", step_dict, "info_diff_flan_implementation")
    yield metadata

@pytest.fixture
def pg_en_content_blocks(metadata_perfume_genius):
    en_blocks = load_artifact_with_step_name(metadata_perfume_genius, "step_get_en_content_blocks")
    yield en_blocks


@pytest.fixture
def pg_fr_content_blocks(metadata_perfume_genius):
    fr_blocks = load_artifact_with_step_name(metadata_perfume_genius, "step_get_fr_content_blocks")
    yield fr_blocks

@pytest.fixture
def pg_info_gap(metadata_perfume_genius):
    info_gap = load_artifact_with_step_name(metadata_perfume_genius, "step_reasoning_intersection_label")
    yield info_gap

@pytest.fixture
def pg_pronoun(metadata_perfume_genius):
    pronoun = load_artifact_with_step_name(metadata_perfume_genius, "step_infer_pronoun")
    yield pronoun

def test_step_obtain_paragraphs_associations(metadata_perfume_genius, pg_en_content_blocks, pg_fr_content_blocks, pg_info_gap, 
                                             pg_pronoun):
    prev_associations = load_artifact_with_step_name(metadata_perfume_genius, "step_add_fact_to_sent_alignment_info")
    most_mapped_sent_freq_prev_en = prev_associations[0]['aligned_sentence'].value_counts()['counts'].max()
    most_mapped_sent_freq_prev_fr = prev_associations[1]['aligned_sentence'].value_counts()['counts'].max()

    associations = step_forced_align_facts_to_paragraph(pg_info_gap, pg_en_content_blocks, pg_fr_content_blocks, pg_pronoun)
    most_mapped_sent_freq_curr_en = associations[0]['aligned_sentence'].value_counts()['counts'].max()
    most_mapped_sent_freq_curr_fr = associations[1]['aligned_sentence'].value_counts()['counts'].max()
    assert most_mapped_sent_freq_curr_en <= most_mapped_sent_freq_prev_en, ipdb.set_trace()
    assert most_mapped_sent_freq_curr_fr <= most_mapped_sent_freq_prev_fr, ipdb.set_trace()