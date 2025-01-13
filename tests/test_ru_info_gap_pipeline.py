import ipdb
import pytest
from collections import OrderedDict

from flowmason import conduct, SingletonStep, load_artifact, load_artifact, load_artifact_with_step_name, MapReduceStep

from packages.steps.map_dicts import get_en_ru_gpt_info_diff_map_dict, get_caa_map_dict_gpt, get_caa_map_dict_flan_ru
from packages.steps.reductions import reduce_info_gaps, reduce_caa_classifications

@pytest.fixture
def metadata_tim_cook():
    en_bio_id = "Tim_Cook"
    ru_bio_id = "Кук,_Тим"
    step_dict = get_en_ru_gpt_info_diff_map_dict(en_bio_id, ru_bio_id, 
                                                  person_name="Tim Cook", ru_person_name="Тим Кук")
    metadata = conduct("test_cache", step_dict, "info_diff_flan_implementation")
    yield metadata

@pytest.fixture
def ru_facts(metadata_tim_cook):
    ru_facts = load_artifact_with_step_name(metadata_tim_cook, 'step_generate_facts_ru')
    yield ru_facts

@pytest.fixture
def ru_info_gap(metadata_tim_cook):
    info_gaps = load_artifact_with_step_name(metadata_tim_cook, 'step_add_fact_to_sent_alignment_info')
    yield info_gaps

def test_ru_fact_decomposition(ru_facts):
    ru_facts = ru_facts
    assert len(ru_facts) > 0
    for fact_block in ru_facts:
        assert len(fact_block.facts) > 0

def test_ru_info_gaps(ru_info_gap):
    pass

def test_ru_connotations(metadata_tim_cook_w_connotations):
    
    ipdb.set_trace()