import polars as pl
import ipdb
import pytest
from flowmason import conduct, SingletonStep, load_artifact_with_step_name
from packages.steps.map_dicts import get_en_fr_info_diff_map_dict, get_en_caa_classification_map_dict

@pytest.fixture
def metadata_gabriel_attal_info_gap():
    bio_id = "Gabriel_Attal"
    step_dict = get_en_fr_info_diff_map_dict(bio_id, bio_id)
    metadata = conduct("test_cache", step_dict, "info_diff_implementation_test_logs")
    yield metadata

@pytest.fixture
def attal_en_fr_info_gap(metadata_gabriel_attal_info_gap):
    return load_artifact_with_step_name(metadata_gabriel_attal_info_gap, "step_add_fact_to_sent_alignment_info")

@pytest.fixture
def metadata_en_caa_classifications(attal_en_fr_info_gap):
    step_dict = get_en_caa_classification_map_dict(attal_en_fr_info_gap)
    metadata_en_caa_classifications = conduct("test_cache", step_dict, "caa_implementation_test_logs")
    return metadata_en_caa_classifications

@pytest.fixture
def en_caa_classifications(metadata_en_caa_classifications):
    return load_artifact_with_step_name(metadata_en_caa_classifications, "step_contextual_affective_classification").filter(pl.col('language') == 'en')

@pytest.fixture
def fr_caa_classifications(metadata_en_caa_classifications):
    return load_artifact_with_step_name(metadata_en_caa_classifications, "step_contextual_affective_classification").filter(pl.col('language') == 'fr')

def test_contextual_affective_classification_size(en_caa_classifications, attal_en_fr_info_gap, fr_caa_classifications):
    en_info_gap_w_classifications = en_caa_classifications.join(attal_en_fr_info_gap[0], on=['person_name', 'fact_index'], how='inner') 
    assert len(en_info_gap_w_classifications) == len(attal_en_fr_info_gap[0])

    fr_info_gap_w_classifications = fr_caa_classifications.join(attal_en_fr_info_gap[1], on=['person_name', 'fact_index'], how='inner')
    assert len(fr_info_gap_w_classifications) == len(attal_en_fr_info_gap[1])

    eig_unique = en_info_gap_w_classifications.unique('context')
    fig_unique = fr_info_gap_w_classifications.unique('context')
    ipdb.set_trace()
    # TODO: pass this test