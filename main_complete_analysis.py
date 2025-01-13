import loguru
import numpy as np
import ipdb
import polars as pl
from collections import OrderedDict
from flowmason import conduct, SingletonStep, load_artifact_with_step_name, MapReduceStep, load_mr_artifact
import click
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.dummy import DummyClassifier
from tqdm import tqdm
import torch

from packages.steps.info_diff_steps import BioFilenotFoundError, ExceptionOOMSingleDataPoint
from packages.steps.map_dicts import get_en_fr_info_diff_map_dict, get_en_fr_info_diff_map_dict_flan , get_caa_map_dict_fr, get_en_ru_gpt_info_diff_map_dict, get_caa_map_dict_gpt,\
    get_en_ru_info_diff_map_dict_flan, get_caa_map_dict_flan_ru, get_en_fr_ablation_dict,\
    get_caa_map_dict_fr_gpt
from packages.steps.reductions import reduce_info_gaps, reduce_caa_classifications, reduce_paragraph_ablation
from packages.steps.caa_steps import step_prep_for_caa, step_caa_multi_sentence, step_caa_multi_sentence_flan, InfoGapEmptyError, NoPronounError
from packages.annotate import annotate_frame, load_save_if_nexists
from packages.flan_query import ask_flan_about_fact_intersection, ask_mt5_about_fact_intersection
# from packages.constants import ANNOTATION_SAVE_PATH, NUM_CONTEXT_SRC, NUM_CONTEXT_TGT, NUM_RETRIEVALS
from packages.constants import NUM_CONTEXT_SRC, NUM_CONTEXT_TGT, NUM_RETRIEVALS, SCRATCH_DIR, CURRENT_EN_BIO_IDS, CURRENT_FR_BIO_IDS, CURRENT_PERSON_NAMES, ANNOTATION_SAVE_PATH, EN_FR_BIO_NAME_CSVS,\
    EN_RU_BIO_NAME_CSV

logger = loguru.logger
def step_prep_annotation_frame(info_gap_dfs, tgt_lang_code, intersection_label, **kwargs) -> pl.DataFrame:
    en_info_gap_df = info_gap_dfs[0]
    tgt_info_gap_df = info_gap_dfs[1]
    
    def get_annotation_frame(src_info_gap_df, tgt_info_gap_df):
        src_info_annotation_rows = []
        people_names = en_info_gap_df['person_name'].unique()
        num_facts_to_sample = 10
        for person_name in people_names:
            paragraph_indices = src_info_gap_df.filter(pl.col('person_name') == person_name)['paragraph_index'].unique().to_list()
            # sample a paragraph and then sample a fact from that paragraph until we hit num_facts_to_sample.
            num_sampled = 0
            while num_sampled < num_facts_to_sample:
                paragraph_index = np.random.choice(paragraph_indices)
                # sample a fact from the paragraph
                fact_df = src_info_gap_df.filter(pl.col('person_name') == person_name).filter(pl.col('paragraph_index') == paragraph_index)\
                    .sample(n=1, with_replacement=False)\
                    .select(['fact', 'fact_index', 'paragraph_index', 'person_name', 'info_retrieval_mapping', intersection_label])
                src_fact_index = fact_df['fact_index'].to_list()[0]
                retrieval_mapping = fact_df['info_retrieval_mapping'].to_list()[0]
                # TODO: need to sort here to make sure we take most similar.
                tgt_fact_indices = [
                    index for index, value in sorted(retrieval_mapping, key = lambda x: x[1], reverse=True)
                ][:NUM_RETRIEVALS]  # need to take the top 2 here since that's what we use for the prediction?
                tgt_contexts = []
                for tgt_fact_index in tgt_fact_indices:
                    tgt_contexts.append(
                        tgt_info_gap_df.filter((pl.col('fact_index') <= tgt_fact_index)\
                                            & (pl.col('person_name')==person_name))['fact'].to_list()[-NUM_CONTEXT_TGT:]
                    )
                src_context = src_info_gap_df.filter((pl.col('fact_index') <= src_fact_index)\
                                                    & (pl.col('person_name') == person_name))['fact'].to_list()[-NUM_CONTEXT_SRC:]
                # add the src_context and tg_contexts to the fact_df
                fact_df = fact_df.with_columns(
                    pl.Series(name='src_context', values=[src_context]),
                    pl.Series(name='tgt_contexts', values=[tgt_contexts])
                )
                # TODO: we need the context for the src fact and the target facts.

                # TODO: make sure to add the person name here
                # src_info_gap_df.filter((pl.col('paragraph_index') == src_paragraph_index) & (pl.col('fact_index') <= fact_index))['fact'].to_list()[-3:]

                # TODO: figure out what columns we need to select.
                src_info_annotation_rows.append(
                    fact_df.select(['fact', 'fact_index', 'person_name', 'src_context', 'tgt_contexts', 'paragraph_index', intersection_label])
                )
                num_sampled += 1
        return pl.concat(src_info_annotation_rows)
    en_annotation_frame = get_annotation_frame(en_info_gap_df, tgt_info_gap_df).with_columns([pl.lit('en').alias('language')])
    tgt_annotation_frame = get_annotation_frame(tgt_info_gap_df, en_info_gap_df).with_columns([pl.lit(tgt_lang_code).alias('language')])
    return pl.concat([en_annotation_frame, tgt_annotation_frame]).sample(fraction=1.0, with_replacement=False, shuffle=True)

def step_annotate_complete_tgt(annotation_frame: pl.DataFrame, 
                               **kwargs):
    # get today's date in form MM-DD
    today_str= "03-06"
    # today = datetime.today()

    annotation_frame = load_save_if_nexists(annotation_frame, f"{ANNOTATION_SAVE_PATH}/annotation_{today_str}.json")
    def ask_question(fact_row):
        context_str = fact_row['src_context']
        candidate_tgt_contexts = fact_row['tgt_contexts']
        tgt_contexts_str = '\n'.join([str(context) for context in candidate_tgt_contexts])
        tgt_lang = 'French' if fact_row['language'] == 'en' else 'English'
        question = f"\n\nConsider the following fact(s) about {fact_row['person_name']}:\n\n{context_str}\n\nIs the final fact present in the {tgt_lang} Wikipedia article about {fact_row['person_name']}?\n\nHere are some snippets from the {tgt_lang} article:\n {tgt_contexts_str}\n\n( yesa / yesr / no ): "
        return question

    annotated_frame = annotate_frame(
        annotation_frame, # frame containing data to annotate 
        num_samples=10, # number of samples to annotate in one setting
        annotation_columns=['fact_in_tgt'], # column to store the annotation in
        question_fns=[ask_question],
        answer_validate_fn=[lambda answer: answer.lower() in ['yesa', 'yesr', 'no']],
    )
    annotated_frame.write_json(f"{ANNOTATION_SAVE_PATH}/annotation_{today_str}.json")
    return

@click.command()
def execute_complete_gpt():
    full_map_dict = OrderedDict()
    info_gap_map_dict = get_en_fr_info_diff_map_dict()
    caa_map_dict = get_caa_map_dict_fr_gpt()

    # reduce_info_gaps,
    # 'fr_bio_id',
    # [BioFilenotFoundError, NoPronounError, ExceptionOOMSingleDataPoint, np.AxisError]

    # TODO: need to watch out for the Abdellah bio since its french wikipedia page is a little weird 
    # TODO: need to log the total number of tokens for the queries somewhere/somehow
        # reduce_info_gaps,
        # 'fr_bio_id',
        # [BioFilenotFoundError, NoPronounError, ExceptionOOMSingleDataPoint, np.AxisError]
    full_map_dict['map_step_compute_info_gap'] = MapReduceStep(info_gap_map_dict, 
        {
            'en_bio_id': ["Gabriel_Attal"],
            'fr_bio_id': ["Gabriel_Attal"], 
            'person_name': ["Gabriel Attal"],
            'tgt_person_name': ["Gabriel Attal"]
        },{
        'version': '001'
        }, 
        reduce_info_gaps, 
        'fr_bio_id',
        [BioFilenotFoundError, NoPronounError, ExceptionOOMSingleDataPoint, np.AxisError]
    )
    full_map_dict['map_step_compute_connotations'] = MapReduceStep(caa_map_dict,
        {
            'en_bio_id': ["Gabriel_Attal"],
            'fr_bio_id': ["Gabriel_Attal"],
            'person_name': ["Gabriel_Attal"]
        },{
            'version': '001', 
            'map_en_fr_info_gaps': 'map_step_compute_info_gap'
        },
        reduce_caa_classifications,
        'fr_bio_id', 
        [InfoGapEmptyError]
    )
    # full_map_dict['prep_for_caa'] = SingletonStep(step_prep_for_caa, {
    #     'en_fr_info_gaps': 'map_step_compute_info_gap',
    #     'version': '004'
    # })
    # full_map_dict['step_compute_caa_multi_sentence'] = SingletonStep(step_caa_multi_sentence, {
    #     'en_fr_info_gaps': 'prep_for_caa',
    #     'version': '002'
    # })
    # full_map_dict['step_prep_annotation_frame'] = SingletonStep(step_prep_annotation_frame, {
    #     'info_gap_dfs': 'map_step_compute_info_gap', 
    #     'version': '003'
    # })
    # full_map_dict['step_annotate_complete_tgt'] = SingletonStep(step_annotate_complete_tgt, {
    #     'annotation_frame': 'step_prep_annotation_frame',
    #     'version': '001'
    # })
    metadata = conduct(os.path.join(SCRATCH_DIR, "full_cache"), full_map_dict, "full_analysis_logs")
    info_gap_dfs = load_mr_artifact(metadata[0])
    connotation_dfs = load_mr_artifact(metadata[-1])
    ipdb.set_trace()

def _parse_response(response_raw):
    assert ',' in response_raw, f"Response does not contain a comma: {response_raw}"
    response_label = response_raw[1:response_raw.index(',')]
    return response_label



@click.command()
@click.option('--start_index', default=0, help='The index to start at')
@click.option('--end_index', default=0, help='The index to start at')
def execute_complete_flan(start_index: int, end_index: int):
    full_map_dict = OrderedDict()
    info_gap_map_dict = get_en_fr_info_diff_map_dict_flan()
    caa_map_dict = get_caa_map_dict_fr()

    bio_id_name_frame = pl.read_csv(EN_FR_BIO_NAME_CSVS) 
    if end_index == 0:
        end_index = len(bio_id_name_frame)
    bio_id_name_frame = bio_id_name_frame[start_index:end_index]

    en_bio_ids = bio_id_name_frame['en_bio_id'].to_list()
    fr_bio_ids = bio_id_name_frame['fr_bio_id'].to_list()
    person_names = bio_id_name_frame['name'].to_list()

    full_map_dict['map_step_compute_info_gap'] = MapReduceStep(info_gap_map_dict,
        {
            'en_bio_id': en_bio_ids,
            'fr_bio_id': fr_bio_ids,
            'person_name': person_names 
        },{
        'version': '001'
        }, 
        reduce_info_gaps,
        'fr_bio_id',
        [BioFilenotFoundError, NoPronounError, ExceptionOOMSingleDataPoint, np.AxisError]
    )
    full_map_dict['map_step_compute_connotations'] = MapReduceStep(caa_map_dict,
        {
            'en_bio_id': en_bio_ids,
            'fr_bio_id': fr_bio_ids,
            'person_name': person_names 
        },{
            'version': '001', 
            'map_en_fr_info_gaps': 'map_step_compute_info_gap'
        },
        reduce_caa_classifications,
        'fr_bio_id', 
        [InfoGapEmptyError]
    )
    metadata = conduct(os.path.join(SCRATCH_DIR, "full_cache_flan"), full_map_dict, "full_analysis_logs")
    connotation_df = load_artifact_with_step_name(metadata, "map_step_compute_connotations")
    info_gap_df = load_artifact_with_step_name(metadata, "map_step_compute_info_gap")
    ipdb.set_trace()
    # joint_frame_en = en_connotations_error_filtered.explode('fact_index').join(en_info_gap, on=['fact_index', 'person_name'], how='inner').select(['caa_response_parsed', 'person_name', 'fact_index', 'fact', 'gpt-4_intersection_label'])
    # pos_frame_en = joint_frame_en.filter(pl.col('caa_response_parsed')=='positive')
    # print('Positive Connotations')
    # print(pos_frame_en.with_columns(pl.col(ig_label).count().over('person_name').alias('num_facts_total')).group_by('person_name', ig_label).agg(pl.count('fact'), pl.first('num_facts_total'))\
    #     .select(pl.col('person_name'), pl.col(ig_label), (pl.col('fact') / pl.col('num_facts_total')).alias('percent'), pl.col('fact'), pl.col('num_facts_total')).sort('person_name').to_pandas().to_markdown())

    # print('Negative Connotations')
    # neg_frame_en = joint_frame_en.filter(pl.col('caa_response_parsed')=='negative')
    # print(neg_frame_en.with_columns(pl.col(ig_label).count().over('person_name').alias('num_facts_total')).group_by('person_name', ig_label).agg(pl.count('fact'), pl.first('num_facts_total'))\
    #     .select(pl.col('person_name'), pl.col(ig_label), (pl.col('fact') / pl.col('num_facts_total')).alias('percent'), pl.col('fact'), pl.col('num_facts_total')).sort('person_name').to_pandas().to_markdown())

@click.command()
@click.option('--start_index', default=0, help='The index to start at')
@click.option('--end_index', default=0, help='The index to start at')
def execute_complete_mt5_en_ru(start_index: int, end_index: int):
    bio_id_name_frame = pl.read_csv(EN_RU_BIO_NAME_CSV, separator='\t')
    if end_index == 0:
        end_index = len(bio_id_name_frame)
    bio_id_name_frame = bio_id_name_frame[start_index:end_index]

    en_bio_ids = bio_id_name_frame['en_bio_id'].to_list()
    ru_bio_ids = bio_id_name_frame['ru_bio_id'].to_list()
    en_person_names = bio_id_name_frame['en_name'].to_list()
    ru_person_names = bio_id_name_frame['ru_name'].to_list()

    full_map_dict = OrderedDict()
    info_gap_map_dict = get_en_ru_info_diff_map_dict_flan()
    caa_map_dict = get_caa_map_dict_flan_ru()
    full_map_dict['map_step_compute_info_gap'] = MapReduceStep(info_gap_map_dict,
        {
            'en_bio_id': en_bio_ids,
            'tgt_bio_id': ru_bio_ids,
            'ru_bio_id': ru_bio_ids, 
            'person_name': en_person_names,
            'tgt_person_name': ru_person_names 
        },{
        'version': '001'
        }, 
        reduce_info_gaps,
        'ru_bio_id', 
        [np.AxisError, BioFilenotFoundError]
    )
    full_map_dict['map_step_compute_caa'] = MapReduceStep(caa_map_dict,
        {
            'en_bio_id': en_bio_ids,
            'tgt_bio_id': ru_bio_ids,
            'ru_bio_id': ru_bio_ids,
            'person_name': en_person_names,
            'tgt_person_name':ru_person_names 
        }, {
            'version': '001', 
            'map_en_tgt_info_gaps': 'map_step_compute_info_gap'
        },
        reduce_caa_classifications,
        'ru_bio_id',
        [InfoGapEmptyError]
    )
    metadata = conduct(os.path.join(SCRATCH_DIR, "full_cache_mt5_en_ru"), full_map_dict, "en_ru_mt5_logs")

@click.command()
def execute_complete_gpt_en_ru():
    info_gap_map_dict = get_en_ru_gpt_info_diff_map_dict()
    caa_map_dict = get_caa_map_dict_gpt()
    full_map_dict = OrderedDict()
    # en_bio_ids = ['Pyotr_Ilyich_Tchaikovsky', 'Tim_Cook', 'Dmitry_Kuzmin', 'Masha_Gessen', 'Nikolay_Alexeyev', 'James_Baldwin', 'Ali_Feruz', 'Elena_Kostyuchenko', 'Mikhail_Zygar', 'Pyotr_Verzilov', 'Sergey_Sosedov', 'Yekaterina_Samutsevich']
    # ru_bio_ids = ['Чайковский,_Пётр_Ильич', 'Кук,_Тим', 'Кузьмин,_Дмитрий_Владимирович', 'Гессен,_Мария_Александровна', 'Алексеев,_Николай_Александрович_(активист)', 'Болдуин,_Джеймс', 'Дело_Али_Феруза', 'Костюченко,_Елена_Геннадьевна', 'Зыгарь,_Михаил_Викторович', 'Верзилов,_Пётр_Юрьевич', 'Соседов,_Сергей_Васильевич', 'Самуцевич,_Екатерина_Станиславовна']
    # en_names = ['Pyotr Ilyich Tchaikovsky', 'Tim Cook', 'Dmitry Kuzmin', 'Masha Gessen', 'Nikolay Alexeyev', 'James Baldwin', 'Ali Feruz', 'Elena Kostyuchenko', 'Mikhail Zygar', 'Pyotr Verzilov', 'Sergey Sosedov', 'Yekaterina Samutsevich']
    # ru_names =  ['Пётр Ильич Чайковский', 'Тим Кук', 'Кузьмин, Дмитрий Владимирович', 'Мария Александровна Гессен', 'Николай Александрович Алексеев', 'Джеймс Болдуин', 'Али Феруз', 'Еле́на Генна́дьевна Костюче́нко', 'Михаи́л Ви́кторович Зы́гарь', 'Пётр Ю́рьевич Верзилов', 'Серге́й Васи́льевич Сосе́дов', 'Екатери́на Станисла́вовна Самуце́вич']
    en_bio_ids = ['Tim_Cook']
    ru_bio_ids = ['Кук,_Тим']
    en_names = [ 'Tim Cook']
    ru_names =  ['Тим Кук']
    full_map_dict['map_step_compute_info_gap'] = MapReduceStep(info_gap_map_dict,
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
        [InfoGapEmptyError]
    )
    full_map_dict['prep_annotation_frame'] = SingletonStep(step_prep_annotation_frame, {
        'info_gap_dfs': 'map_step_compute_info_gap',
        'tgt_lang_code': 'ru',
        'version': '002', 
        'intersection_label': 'gpt-4_intersection_label'
    })
    full_map_dict['map_step_compute_caa'] = MapReduceStep(caa_map_dict,
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
    # Next task: get this to launch.
    metadata = conduct(os.path.join(SCRATCH_DIR, "full_cache_gpt_en_ru"), full_map_dict, "en_ru_gpt_logs") 
    info_gap_frames = load_mr_artifact(metadata[0])
    connotation_frame = load_mr_artifact(metadata[-1])
    annotation_frame = load_artifact_with_step_name(metadata, "prep_annotation_frame")
    ipdb.set_trace()

@click.command()
def execute_paragraph_align_ablation():

    def step_compute_metrics(annotation_frame: pl.DataFrame, info_gap_df: pl.DataFrame, **kwargs):
        """
        """
        assert len(annotation_frame) == len(info_gap_df), ipdb.set_trace()
        def add_query_fact(frame):
            return frame.with_columns([
                pl.col('src_context').list.last().alias('query_fact')
            ])

        def map_to_binary_annotations(frame, annotation_column):
            annotator = annotation_column.split('_')[0]
            frame = frame.with_columns([pl.when(pl.col(annotation_column).is_in(['A', 'B', 'C', 'D']))\
                .then('yes')\
                .otherwise('no').alias(f'fact_in_tgt_{annotator}')
            ])
            return frame
            
        info_gap_df = add_query_fact(info_gap_df)
        annotation_frame = add_query_fact(annotation_frame)
        annotator = 'samir'
        annotation_frame = map_to_binary_annotations(annotation_frame, f'{annotator}_annotations')
        metric_frame = annotation_frame.join(info_gap_df, on=['query_fact', 'person_name'], how='inner')
        
        assert len(metric_frame) == len(annotation_frame), ipdb.set_trace()
        # Need to preprocess the annotation column from A,B,C,D,E to yes or no. Then, compute the classification report
        en_metric_frame = metric_frame.filter(pl.col('language') == 'en') 
        print(classification_report(en_metric_frame[f'fact_in_tgt_{annotator}'], en_metric_frame['gpt-4_paragraph_abln_label']))
        
        fr_metric_frame = metric_frame.filter(pl.col('language') == 'fr') 
        print(classification_report(fr_metric_frame[f'fact_in_tgt_{annotator}'], fr_metric_frame['gpt-4_paragraph_abln_label']))



    person_names = ['Sophie Labelle', 'Gabriel Attal', 'Caroline Mécary', 'Tim Cook', 'Ellen DeGeneres', 'Kim Petras', 'Alan Turing', 'Philippe Besson', 'Frédéric Mitterrand', 'Abdellah Taïa']
    en_bio_ids = ['Sophie_Labelle', 'Gabriel_Attal', 'Caroline_Mécary', 'Tim_Cook', 'Ellen_DeGeneres', 'Kim_Petras', 'Alan_Turing', 'Philippe_Besson', 'Frédéric_Mitterrand', 'Abdellah_Taïa']
    fr_bio_ids = ['Sophie_Labelle', 'Gabriel_Attal', 'Caroline_Mécary', 'Tim_Cook', 'Ellen_DeGeneres', 'Kim_Petras', 'Alan_Turing', 'Philippe_Besson', 'Frédéric_Mitterrand', 'Abdellah_Taïa']
    step_dict = OrderedDict()
    map_dict = get_en_fr_ablation_dict()
    # TODO: add en and fr bio IDs
    def step_load_annotation_frame(**kwargs):
        annotation_frame = pl.read_json(f"scratch/ethics_annotation_save/vered_annotations_05-20_complete.json")
        return annotation_frame
    step_dict['load_annotation_frame'] = SingletonStep(step_load_annotation_frame, {
        'version': '001'
    })
    step_dict['map_step_compute_info_gap'] = MapReduceStep(map_dict, 
        {
            'person_name': person_names,
            'en_bio_id': en_bio_ids,
            'fr_bio_id': fr_bio_ids
        },{
            'version': '001', 
            'annotated_frame': 'load_annotation_frame'
        }, 
        reduce_paragraph_ablation,
        'fr_bio_id',
        []
    )
    step_dict['compute_metrics_abln'] = SingletonStep(step_compute_metrics, {
        'info_gap_df': 'map_step_compute_info_gap',
        'annotation_frame': 'load_annotation_frame',
        'version': '001'
    })
    metadata = conduct(os.path.join(SCRATCH_DIR, "full_cache_ablation"), step_dict, "ablation_logs")


    ipdb.set_trace()

@click.command()
@click.argument('language')
def execute_entailment_baseline(language):
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-roberta-base')
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-roberta-base')
    #annotation_frame = pl.read_json(f"scratch/ethics_annotation_save/en_ru_prepped_annotations.json")

    def add_language_to_frame(ig_frame, language):
        return ig_frame.with_columns([pl.lit(language).alias('language')])
    if language == 'fr':
        fr_info_gap = pl.read_json("fr_info_gap.json").drop('en_bio_id')
        en_info_gap = pl.read_json("en_info_gap.json").drop('fr_bio_id')
        annotation_frame = pl.read_json(f"scratch/ethics_annotation_save/vered_annotations_05-20_complete.json")
        prediction_column = 'gpt-4_intersection_label'
    elif language == 'ru':
        ru_info_gap = pl.read_json('annotations_with_predictions_en_ru.json').filter(pl.col('language') == 'ru')
        en_info_gap = pl.read_json('annotations_with_predictions_en_ru.json').filter(pl.col('language') == 'en')
        annotation_frame = pl.read_json('annotations_with_predictions_en_ru.json')
        prediction_column = 'gpt4v_intersection_label'
    else:
        raise ValueError("Language must be either 'fr' or 'ru'")
    label_mapping = ['contradiction', 'entailment', 'neutral']

    def add_predictions_to_annotation_frame(annot_frame, prediction_column, language):
        en_annot_frame = annot_frame.filter(pl.col('language')== 'en')
        info_gap_columns = [prediction_column, 'person_name', 'fact_index']
        en_annot_frame = en_annot_frame.join(en_info_gap.select(info_gap_columns), 
                                             on=['person_name', 'fact_index'], 
                                             how='inner')

        if language == 'fr':
            fr_annot_frame = annot_frame.filter(pl.col('language') == 'fr')
            fr_annot_frame = fr_annot_frame.join(fr_info_gap.select(info_gap_columns), 
                                                on=['person_name', 'fact_index'], 
                                                how='inner')
        
            return pl.concat([en_annot_frame, fr_annot_frame])
        elif language == 'ru':
            ru_annot_frame = annot_frame.filter(pl.col('language') == 'ru')
            ru_annot_frame = ru_annot_frame.join(ru_info_gap.select(info_gap_columns), 
                                                on=['person_name', 'fact_index'], 
                                                how='inner')
            return pl.concat([en_annot_frame, ru_annot_frame])
    annotation_frame = add_predictions_to_annotation_frame(annotation_frame, prediction_column, language)

    pbar = tqdm(total=len(annotation_frame))
    def classifier_bootstrap_test(predictions_one, 
                                predictions_two, 
                                ground_truth, num_samples=1000):
        num_clf_one_better = 0
        predictions_one = np.array(predictions_one)
        predictions_two = np.array(predictions_two)
        ground_truth = np.array(ground_truth)
        for _ in range(num_samples):
            indices = np.random.choice(len(predictions_one), len(predictions_one), replace=True)
            clf_one = classification_report(ground_truth[indices], 
                                            predictions_one[indices], 
                                            output_dict=True)
            # check if all the ground truth indices are "yes"
            if np.all(ground_truth[indices] == 'yes'):
                ipdb.set_trace()
            clf_two = classification_report(ground_truth[indices], 
                                            predictions_two[indices], 
                                            output_dict=True)
            if clf_one['macro avg']['f1-score'] > clf_two['macro avg']['f1-score']:
                num_clf_one_better += 1
        logger.info(f"Classifier one was better in {num_clf_one_better} out of {num_samples} samples (p-value: {num_clf_one_better/num_samples})")
    def _predict_entailment(row):
        premises = row['src_contexts_en_translation'] # List[str]
        hypothesis_groups = row['tgt_contexts_en_translation'] # List[List[str]]
        query_fact = premises[0]
        tgt_facts = [hypothesis_grp[-1] for hypothesis_grp in hypothesis_groups]
        tgt_predictions = []
        for tgt_fact in tgt_facts:
            inputs = tokenizer([tgt_fact], [query_fact], return_tensors="pt")
            logits = model(**inputs)[0] # entailment, neutral, contradiction
            prediction = label_mapping[logits.argmax().item()]
            tgt_predictions.append(prediction)
        pbar.update(1)
        if 'entailment' in tgt_predictions:
            return 'yes'
        else:
            return 'no'

    def map_to_binary_annotations(frame, annotation_column):
        annotator = annotation_column.split('_')[0]
        frame = frame.with_columns([pl.when(pl.col(annotation_column).is_in(['A', 'B', 'C', 'D']))\
            .then('yes')\
            .otherwise('no').alias(f'fact_in_tgt_{annotator}')
        ])
        return frame
    # add 'fact_in_tgt_samir' to the annotation frame
    annotation_frame = map_to_binary_annotations(annotation_frame, 'samir_annotations')
    annotation_frame = annotation_frame.with_columns([
        pl.struct(['src_contexts_en_translation',
                     'tgt_contexts_en_translation'])\
                      .apply(_predict_entailment).alias('entailment_prediction')
    ])
    # classifier_bootstrap_test(annotation_frame[prediction_column].to_list(),
    #                             annotation_frame['entailment_prediction'].to_list(),
    #                             annotation_frame['fact_in_tgt_samir'].to_list())
    # print the classification reports for language == 'en' and for language == 'fr'
    en_annotation_frame = annotation_frame.filter(pl.col('language') == 'en')
    fr_annotation_frame = annotation_frame.filter(pl.col('language') == language)
    # print(classification_report(en_annotation_frame['fact_in_tgt_samir'], en_annotation_frame['entailment_prediction']))
    # print(classification_report(fr_annotation_frame['fact_in_tgt_samir'], fr_annotation_frame['entailment_prediction']))
    classifier_bootstrap_test(annotation_frame[prediction_column].to_list(),
                                annotation_frame['entailment_prediction'].to_list(),
                                annotation_frame['fact_in_tgt_samir'].to_list())

    dummy_clf = DummyClassifier(strategy='uniform')
    dummy_clf.fit(annotation_frame['entailment_prediction'], annotation_frame['fact_in_tgt_samir'])
    dummy_predictions = dummy_clf.predict(annotation_frame['entailment_prediction'])
    annotation_frame = annotation_frame.with_columns([
        pl.Series(name = 'dummy_predictions', values=dummy_predictions)
    ])
    en_annotation_frame = annotation_frame.filter(pl.col('language') == 'en')
    fr_annotation_frame = annotation_frame.filter(pl.col('language') == language)
    classifier_bootstrap_test(annotation_frame[prediction_column].to_list(),
                                annotation_frame['dummy_predictions'].to_list(),
                                annotation_frame['fact_in_tgt_samir'].to_list())
    # print(classification_report(en_annotation_frame['fact_in_tgt_samir'], en_annotation_frame['dummy_predictions']))
    # print(classification_report(fr_annotation_frame['fact_in_tgt_samir'], fr_annotation_frame['dummy_predictions']))
    # print(classification_report(annotation_frame['fact_in_tgt_samir'], annotation_frame['dummy_predictions']))
    
@click.command()
def assess_flan_on_annotations(language):
    assert language == 'fr' or language == 'ru'
    if language == 'fr':
        generate_prompt, predict_local = ask_flan_about_fact_intersection(f"/Users/samir/ethics-lgbt-artifacts/connotation_flan_t5_info_gap_twp=False/checkpoint-8600")
        annotation_frame = pl.read_json("vered_annotations_05-20_complete.json")
    else:
        path = "/Users/samir/ethics-lgbt-artifacts/connotation_flan_t5_info_gap_twp=False_lang_pair=ru_en_mt5=True_do_sweep=True/checkpoint-5200"
        generate_prompt, predict_local = ask_mt5_about_fact_intersection(path)
        annotation_frame = pl.read_json("en_ru_prepped_annotations.json")

    src_fact_context_lst = annotation_frame['src_context'].to_list()
    tgt_contexts_lst = annotation_frame['tgt_contexts'].to_list()
    language_lst = annotation_frame['language'].to_list()
    person_name_lst = annotation_frame['person_name'].to_list()

    predictions = []

    def map_to_binary_annotations(frame, annotation_column):
        annotator = annotation_column.split('_')[0]
        frame = frame.with_columns([pl.when(pl.col(annotation_column).is_in(['A', 'B', 'C', 'D']))\
            .then('yes')\
            .otherwise('no').alias(f'fact_in_tgt_{annotator}')
        ])
        return frame

    annotation_frame = map_to_binary_annotations(annotation_frame, 'samir_annotations')
    with torch.no_grad():
        result_frame = annotation_frame.with_columns([
            pl.struct(['src_contexts_en_translation', 
                       'tgt_contexts_en_translation'])\
                        .apply(_predict_entailment).alias('entailment_prediction')
        ])
    logger = loguru.logger
    logger.info("Entailment Prediction Results")
    print(classification_report(result_frame['fact_in_tgt_samir'], result_frame['entailment_prediction']))
    logger.info("Model Performance")
    print(classification_report(result_frame['fact_in_tgt_samir'], result_frame[prediction_column]))

    def classifier_bootstrap_test(predictions_one, 
                                  predictions_two, 
                                  ground_truth, num_samples=1000):
        num_clf_one_better = 0
        predictions_one = np.array(predictions_one)
        predictions_two = np.array(predictions_two)
        ground_truth = np.array(ground_truth)
        for _ in range(num_samples):
            indices = np.random.choice(len(predictions_one), len(predictions_one), replace=True)
            clf_one = classification_report(ground_truth[indices], 
                                            predictions_one[indices], 
                                            output_dict=True)
            # check if all the ground truth indices are "yes"
            if np.all(ground_truth[indices] == 'yes'):
                ipdb.set_trace()
            clf_two = classification_report(ground_truth[indices], 
                                            predictions_two[indices], 
                                            output_dict=True)
            if clf_one['macro avg']['f1-score'] > clf_two['macro avg']['f1-score']:
                num_clf_one_better += 1
        logger.info(f"Classifier one was better in {num_clf_one_better} out of {num_samples} samples (p-value: {num_clf_one_better/num_samples})")
    # classifier_bootstrap_test(result_frame['gpt-4_intersection_label'].to_list(), 
    #                           result_frame['entailment_prediction'].to_list(),
    #                           result_frame['fact_in_tgt_samir'].to_list())
        

    # fit a dummy classifier to the data 100 times and pick the best predictions
    all_performances = []
    all_predictions = []
    for _ in range(100):
        dummy_clf = DummyClassifier(strategy='stratified')
        dummy_clf.fit(result_frame['entailment_prediction'], result_frame['fact_in_tgt_samir'])
        dummy_predictions = dummy_clf.predict(result_frame['entailment_prediction'])
        f1 = classification_report(result_frame['fact_in_tgt_samir'], 
                                   dummy_predictions,
                                   output_dict=True)['macro avg']['f1-score']
        all_performances.append(f1)
        all_predictions.append(dummy_predictions)
        # if f1 > best_performance:
        #     best_performance = f1
        #     best_predictions = dummy_predictions
    # get the index of the median f1 score
    median_index = np.argsort(all_performances)[len(all_performances) // 2]
    median_performance = all_performances[median_index]
    median_predictions = all_predictions[median_index]
    logger.info(f"Best Dummy Classifier Performance: {median_performance}")
    classifier_bootstrap_test(result_frame[prediction_column].to_list(),
                                median_predictions,
                                result_frame['fact_in_tgt_samir'].to_list())
    # classifier_bootstrap_test(result_frame['gpt-4_intersection_label'].to_list(),
    #                             dummy_predictions,
    #                             result_frame['fact_in_tgt_samir'].to_list())
    ipdb.set_trace()


@click.group()
def main():
    pass

# connotation_df_en_ru_gpt_2024_05_30.json
main.add_command(execute_complete_gpt)
main.add_command(execute_complete_flan)
main.add_command(execute_complete_gpt_en_ru)
main.add_command(execute_complete_mt5_en_ru)
main.add_command(execute_paragraph_align_ablation)
main.add_command(execute_entailment_baseline)
main.add_command(assess_flan_on_annotations)

if __name__ == '__main__':
    main()