import pytest
from flowmason import conduct, load_artifact, load_artifact_with_step_name
import ipdb
from wikipedia_edit_scrape_tool import get_text, Header, Paragraph 

from packages.steps.map_dicts import get_en_fr_info_diff_map_dict_flan
from packages.steps.info_diff_steps import step_obtain_paragraphs_associations

@pytest.fixture
def metadata_sophie():
    fr_bio_id = "Stephanie_Beatriz"
    en_bio_id = "Stephanie_Beatriz"
    step_dict = get_en_fr_info_diff_map_dict_flan(en_bio_id, fr_bio_id, person_name="Stephanie Beatriz")
    metadata = conduct("test_cache", step_dict, "info_diff_flan_implementation")
    yield metadata

@pytest.fixture
def metadata_perfume_genius():
    en_bio_id = "Perfume_Genius"
    fr_bio_id = "Perfume_Genius"
    step_dict = get_en_fr_info_diff_map_dict_flan(en_bio_id, fr_bio_id, person_name="Perfume Genius")
    metadata = conduct("test_cache", step_dict, "info_diff_flan_implementation")
    yield metadata

@pytest.fixture
def metadata_helen_stephens():
    fr_bio_id = "Helen_Stephens"
    en_bio_id = "Helen_Stephens"
    step_dict = get_en_fr_info_diff_map_dict_flan(en_bio_id, fr_bio_id, person_name="Helen Stephens")
    # only take the first 4 steps of the ordered dict
    # step_dict = {k: v for i, (k, v) in enumerate(step_dict.items()) if i < 4}
    metadata = conduct("test_cache", step_dict, "info_diff_flan_implementation")
    yield metadata

@pytest.fixture
def en_paragraphs_helen_stephens(metadata_helen_stephens):
    en_blocks = load_artifact_with_step_name(metadata_helen_stephens, "step_get_en_content_blocks")
    yield [block for block in en_blocks if isinstance(block, Paragraph)]

@pytest.fixture
def fr_paragraphs_helen_stephens(metadata_helen_stephens):
    fr_blocks = load_artifact_with_step_name(metadata_helen_stephens, "step_get_fr_content_blocks")
    yield [block for block in fr_blocks if isinstance(block, Paragraph)]

@pytest.fixture
def fr_fact_blocks_helen_stephens(metadata_helen_stephens):
    fr_fact_blocks = load_artifact_with_step_name(metadata_helen_stephens, "step_generate_facts_fr_flan")
    yield fr_fact_blocks

@pytest.fixture
def en_fact_blocks_helen_stephens(metadata_helen_stephens):
    en_fact_blocks = load_artifact_with_step_name(metadata_helen_stephens, "step_generate_facts_en_flan")
    yield en_fact_blocks

def test_fact_decomposition_helen_stephens(en_paragraphs_helen_stephens, en_fact_blocks_helen_stephens):
    en_paragraphs = en_paragraphs_helen_stephens
    en_fact_blocks = en_fact_blocks_helen_stephens
    assert len(en_paragraphs) == len(en_fact_blocks)
    for fact_block in en_fact_blocks:
        assert len(fact_block.facts) > 0

def test_fact_decomposition_helen_stephens_fr(fr_paragraphs_helen_stephens, fr_fact_blocks_helen_stephens):
    fr_paragraphs = fr_paragraphs_helen_stephens
    fr_fact_blocks = fr_fact_blocks_helen_stephens
    assert len(fr_paragraphs) == len(fr_fact_blocks)
    for fact_block in fr_fact_blocks:
        assert len(fact_block.facts) > 0

def test_step_obtain_paragraphs_associations(en_fact_blocks_helen_stephens, fr_fact_blocks_helen_stephens):
    en_fact_blocks = en_fact_blocks_helen_stephens
    fr_fact_blocks = fr_fact_blocks_helen_stephens
    alignment_dfs = step_obtain_paragraphs_associations('step_paragraph_assoc', '001', en_fact_blocks, fr_fact_blocks)
    ipdb.set_trace()

@pytest.fixture
def get_en_facts(metadata_sophie):
    # TODO: retrieve the English content blocks from the cache using load_artifact.
    en_fact_blocks = load_artifact_with_step_name(metadata_sophie, "step_generate_facts_en_flan")
    yield en_fact_blocks

@pytest.fixture
def en_content_blocks(metadata_sophie):
    en_content_blocks = load_artifact_with_step_name(metadata_sophie, "step_get_en_content_blocks")
    yield en_content_blocks

@pytest.fixture
def info_gap_dfs(metadata_sophie):
    info_gap_dfs = load_artifact_with_step_name(metadata_sophie, "step_reasoning_intersection_label")
    yield info_gap_dfs

def test_en_fact_structures(en_content_blocks, get_en_facts):
    en_content_blocks = en_content_blocks
    en_fact_blocks = get_en_facts
    ipdb.set_trace()
    assert len(list(
        filter(lambda x: isinstance(x, Paragraph), en_content_blocks)
    )) == len(en_fact_blocks)

def test_flan_info_gap(info_gap_dfs):
    info_gap_dfs = info_gap_dfs
    ipdb.set_trace()
    assert len(info_gap_dfs) == 3
