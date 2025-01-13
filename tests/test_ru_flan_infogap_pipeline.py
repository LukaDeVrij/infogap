import dill
from packages.steps.map_dicts import get_en_ru_info_diff_map_dict_flan, get_caa_map_dict_flan_ru
from packages.steps.info_diff_steps import step_obtain_paragraphs_associations
from packages.steps.reductions import reduce_info_gaps, reduce_caa_classifications

import ipdb
from collections import OrderedDict
import pytest
from flowmason import conduct, load_artifact, load_artifact_with_step_name, MapReduceStep

@pytest.fixture
def metadata_tc():
    en_bio_id = "Tim_Cook"
    ru_bio_id = "Кук,_Тим"
    step_dict = get_en_ru_info_diff_map_dict_flan(en_bio_id, ru_bio_id, person_name="Tim Cook", ru_person_name="Тим Кук")
    metadata = conduct("test_cache", step_dict, "info_diff_flan_implementation")
    yield metadata

@pytest.fixture
def map_metadata_tc():
    ig_dict = get_en_ru_info_diff_map_dict_flan()
    caa_dict = get_caa_map_dict_flan_ru()

    en_bio_ids = ['Tim_Cook']
    ru_bio_ids = ['Кук,_Тим']
    en_names = ['Tim Cook']
    ru_names =  ['Тим Кук']
    full_map_dict = OrderedDict()
    full_map_dict['map_step_compute_info_gap'] = MapReduceStep(ig_dict,
        {
            'en_bio_id': en_bio_ids,
            'tgt_bio_id': ru_bio_ids,
            'ru_bio_id': ru_bio_ids, 
            'person_name': en_names,
            'tgt_person_name': ru_names
        },{
        'version': '001'
        }, 
        reduce_info_gaps,
        'ru_bio_id', 
        []
    )
    full_map_dict['map_step_compute_caa'] = MapReduceStep(caa_dict,
        {
            'en_bio_id': en_bio_ids,
            'tgt_bio_id': ru_bio_ids,
            'ru_bio_id': ru_bio_ids,
            'person_name': en_names,
            'tgt_person_name': ru_names
        }, {
            'version': '001', 
            'map_en_tgt_info_gaps': 'map_step_compute_info_gap'
        },
        reduce_caa_classifications,
        'ru_bio_id',
        []
    )
    metadata = conduct("test_cache", full_map_dict, "info_diff_flan_implementation")
    yield metadata

def test_run_map(map_metadata_tc):
    print(map_metadata_tc)
    ig_filename = map_metadata_tc[-1][1]['cache_path']
    with open(ig_filename, 'rb') as f:
        result = dill.load(f)
    ipdb.set_trace()
    # ig_filename = metadata_tc[-1][1]['cache_path']
    # with open(ig_filename, 'rb') as f:
    #     result = dill.load(f)
    # final_ig = result
    # ipdb.set_trace()
    # pass

def test_run(metadata_tc):
    ig_filename = metadata_tc[-1][1]['cache_path']
    with open(ig_filename, 'rb') as f:
        result = dill.load(f)
    final_ig = result
    ipdb.set_trace()
    pass