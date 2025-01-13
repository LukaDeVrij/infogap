import shutil
from typing import Optional
import click
import numpy as np
import wandb
import torch
import evaluate
from tqdm import tqdm
import json
import os
from collections import OrderedDict
from flowmason import conduct, SingletonStep, load_artifact_with_step_name, MapReduceStep
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import polars as pl
import ipdb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, BitsAndBytesConfig
from transformers import MT5ForConditionalGeneration, MT5ForSequenceClassification, T5Tokenizer
from datasets import Dataset, DatasetDict
from functools import partial
from wikipedia_edit_scrape_tool import get_text, Header, Paragraph 
from sklearn.metrics import accuracy_score, f1_score
import loguru

from packages.constants import NUM_CONTEXT_SRC, NUM_CONTEXT_TGT, NUM_RETRIEVALS, SCRATCH_DIR, CONNOTATION_FLAN_SAVE_DIR, HF_CACHE_DIR, FACT_DECOMP_FLAN_SAVE_DIR
from packages.steps.caa_steps import construct_prompt_2_en, construct_prompt_2_fr,\
    _parse_response_gpt, _parse_response_rationale, construct_prompt_2_ru
from packages.steps.info_diff_steps import step_retrieve_prescraped_ru_content_blocks
from packages.gpt_query import construct_fact_intersection_prompt, construct_fact_decomp_prompt

rouge = evaluate.load("rouge")
logger = loguru.logger
# set up wandb
os.environ["WANDB_PROJECT"]="infogap-lgbt"

def step_load_dataset(lang: str, 
                      dataset_path:str, 
                      **kwargs):
    dataset = pl.read_json(dataset_path).filter(pl.col('language')==lang).with_columns(
        pl.struct(['caa_classification', 'caa_query_content'])\
        .map_elements(lambda x: _parse_response_gpt(lang, 
                                                x['caa_query_content'], 
                                                x['caa_classification']))\
        .alias('caa_response_parsed')
    ).filter(pl.col('caa_response_parsed')!='blocked by content filter')
    if 'caa_response_complete' not in dataset.columns:
        dataset = dataset.rename({'caa_classification': 'caa_response_complete'})
    neg_labels = ['neg', 'negatif', 'négatif', 'negative', 'отрицательное', 'отрицательный']
    pos_labels = ['pos', 'positif', 'positive', 'положительный', 'положительное']
    neutral_labels = ['neutral', 'neutre', 'none', 'нейтральный', 'нейтральное', 'Нет']

    if lang == 'en':
        construct_prompt_fn = construct_prompt_2_en 
    elif lang == 'ru':
        construct_prompt_fn = construct_prompt_2_ru
    elif lang == 'fr':
        construct_prompt_fn = construct_prompt_2_fr
    dataset = dataset.with_columns(
        [
        pl.when(pl.col('caa_response_parsed').is_in(neg_labels))\
            .then(pl.lit('neg'))
            .when(pl.col('caa_response_parsed').is_in(pos_labels))
            .then(pl.lit('pos'))
            .when(pl.col('caa_response_parsed').is_in(neutral_labels))
            .then(pl.lit('neutral'))\
            .alias('label'),
        pl.struct(['pronoun', 'caa_query_content', 'person_name']).map_elements(
            lambda x: (construct_prompt_fn(x['caa_query_content'], 
                x['person_name'], 
                x['pronoun'])
            )).alias('prompt')
        ]
    ).with_columns(
        pl.struct(['prompt', 'caa_response_complete'])\
            .map_elements(
                lambda x: _parse_response_rationale(lang, 
                                      x['prompt'],
                                      x['caa_response_complete'])
        ).alias('rationale')
    ).select(['prompt', 'rationale', 'label'])
    return dataset

def step_load_infogap_dataset(src_lang: str, tgt_lang: str, 
                              src_dataset_path: str, 
                              tgt_dataset_path: str, 
                              ig_label: str, **kwargs):
    src_info_gap_df = pl.read_json(src_dataset_path)                                
    tgt_info_gap_df = pl.read_json(tgt_dataset_path)
    ipdb.set_trace() 
    if src_lang == 'en' or src_lang == 'fr':
        person_name_label = 'person_name'
    elif src_lang == 'ru':
        person_name_label = 'ru_person_name'
    else:
        raise ValueError(f"Language {src_lang} not supported.")
    def annotate_llm(src_lang,  src_info_gap_df: pl.DataFrame, tgt_info_gap_df: pl.DataFrame, 
                    person_name: str, paragraph_index,  info_intersection_mapping, fact_index):
        src_paragraph_index = paragraph_index
        src_fact_context = src_info_gap_df.filter((pl.col('paragraph_index') == src_paragraph_index) &\
                                                  (pl.col('fact_index') <= fact_index) &\
                                                  (pl.col(person_name_label)==person_name))['fact'].to_list()[-NUM_CONTEXT_SRC:]
        tgt_contexts = []
        for tgt_index, margin in list(sorted(info_intersection_mapping, key=lambda x: x[1], reverse=True))[:NUM_RETRIEVALS]:
            try:
                tgt_paragraph_index = tgt_info_gap_df.filter((pl.col('fact_index') == tgt_index) &\
                                                             (pl.col(person_name_label)==person_name))['paragraph_index'].to_list()[0]
            except IndexError:
                ipdb.set_trace()
            tgt_context = tgt_info_gap_df.filter((pl.col('paragraph_index') == tgt_paragraph_index) &\
                                                  (pl.col('fact_index') <= tgt_index) &\
                                                  (pl.col(person_name_label) == person_name))['fact'].to_list()[-NUM_CONTEXT_TGT:]
            tgt_contexts.append(tgt_context)
        input_prompt = construct_fact_intersection_prompt(src_lang, tgt_lang, src_fact_context, tgt_contexts, person_name)
        return input_prompt
    annotation_fn = partial(annotate_llm, src_lang, src_info_gap_df, tgt_info_gap_df)
    io_info_gap_df = src_info_gap_df.with_columns([
        pl.struct(['paragraph_index', 'fact_index', 'info_retrieval_mapping', person_name_label]).\
            map_elements(lambda row: annotation_fn(
                row[person_name_label],
                row['paragraph_index'], 
                row['info_retrieval_mapping'], 
                row['fact_index'])).\
            alias(f'info_gap_prompt')
    ]).select(['info_gap_prompt', ig_label, 'paragraph_index', person_name_label])
    assert len(io_info_gap_df) == len(src_info_gap_df)
    ipdb.set_trace()
    # assert that the only values in gpt-4_intersection_label are 'yes' or 'no'
    assert io_info_gap_df[ig_label].is_in(['yes', 'no']).all()
    return io_info_gap_df

def preprocess_function(tokenizer, sample, padding='max_length'):
    model_inputs = tokenizer(sample['prompt']) # don't pad in preprocessing
    label_str = f"({sample['label']}, {sample['rationale']})"
    # json stringifying the label_str
    labels = tokenizer((label_str))
    # if padding == "max_length":
    #     labels["input_ids"] = [
    #         # [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    #         (l if l != tokenizer.pad_token_id else -100) for l in labels["input_ids"]
    #     ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_info_gap(intersection_label, tokenizer, sample):
    model_inputs = tokenizer(sample['info_gap_prompt']) 
    labels = tokenizer(sample[intersection_label])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_fact_decomposition(tokenizer, sample):

    model_inputs = tokenizer(sample['prompt']) 
    labels = tokenizer(sample['facts'])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(tokenizer, eval_preds):
    def _extract_label(datapoint):
        try:
            label = datapoint.split(',')[0][1:]
            return label
            # label_rational_dict = json.loads(tokenizer.decode(datapoint['labels'], skip_special_tokens=True))
            # return label_rational_dict['label']
        except:
            return 'fail'
    preds, labels = eval_preds
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    actual_predictions = [_extract_label(pred) for pred in decoded_preds]
    actual_labels = [_extract_label(label) for label in decoded_labels]
    acc = accuracy_score(actual_labels, actual_predictions)
    print(decoded_preds[:10])
    print(decoded_labels[:10])
    result = {'acc': acc}
    return result

def compute_metrics_info_gap(tokenizer, eval_preds):
    preds, labels = eval_preds
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # check if all the predictions are either 'yes' or 'no'
    if all([pred == 'yes' for pred in decoded_preds]):
        logger.warning("All predictions are 'yes'.")
    elif all([pred == 'no' for pred in decoded_preds]):
        logger.warning("All predictions are 'no'.")
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    f1 = f1_score(decoded_labels, decoded_preds, average='micro')
    print("=====Predictions=====")
    print(decoded_preds[:10])
    print("=====Labels=====")
    print(decoded_labels[:10])
    result = {'micro-f1': f1}
    return result

def compute_metrics_rouge(tokenizer, eval_preds):
    preds, labels = eval_preds
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    # print the first 10 predictions and labels
    print("=====Predictions=====")
    print(decoded_preds[:10])
    print("=====Labels=====")
    print(decoded_labels[:10])
    return rouge_scores

def step_train_model(lang_one_dataset: pl.DataFrame, 
                     clear_save_dir: bool, 
                     langs: str,
                     use_mt5: bool,
                     train_with_peft: bool = False,
                     lang_two_dataset: Optional[pl.DataFrame] = None,
                     **kwargs):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("mps")

    # TODO: add cache dir
    if train_with_peft:
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", cache_dir=HF_CACHE_DIR, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
        )
        model = get_peft_model(model, lora_config)
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    elif use_mt5:
        model = MT5ForConditionalGeneration.from_pretrained("google/mt5-large", cache_dir=HF_CACHE_DIR).to(device)
        tokenizer = AutoTokenizer.from_pretrained("google/mt5-large", cache_dir=HF_CACHE_DIR)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", cache_dir=HF_CACHE_DIR).to(device)
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    if lang_two_dataset is not None:
        dataset = pl.concat([lang_one_dataset, lang_two_dataset])
    else:
        dataset = lang_one_dataset
    hf_dataset = Dataset.from_pandas(dataset.to_pandas()).map(partial(preprocess_function, tokenizer), remove_columns=['prompt', 'rationale', 'label'])
    max_train_length = max([len(x) for x in hf_dataset['input_ids']]) # TODO: shouldn't this be labels instead?
    logger.info(f"Max train length: {max_train_length}")
    hf_dataset = hf_dataset.train_test_split(test_size=0.1)

    # train_dataset, eval_dataset, test_dataset = partition_dataset(dataset)
    output_dir = f"{CONNOTATION_FLAN_SAVE_DIR}_twp={train_with_peft}_langs={langs}_use_mt5={use_mt5}"
    # create output directory if it doesn't exist
    if clear_save_dir:
        shutil.rmtree(output_dir, ignore_errors=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id, 
        padding=True, 
    )
    # TODO: need to make sure that the label for the pad token ID is -100..
    learning_rate = 5e-5 if not use_mt5 else 1e-3
    gradient_accumulation_steps = 1 if not use_mt5 else 4
    training_args = Seq2SeqTrainingArguments(
        report_to="wandb",
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        predict_with_generate=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=False, # Overflows with fp16
        learning_rate=learning_rate,
        num_train_epochs=5,
        generation_max_length=max_train_length,
        # logging & evaluation strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=25,
        auto_find_batch_size=True,
        eval_steps=100,
        save_steps = 200,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        metric_for_best_model="acc",
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=hf_dataset['train'], 
        compute_metrics=partial(compute_metrics, tokenizer),
        eval_dataset=hf_dataset['test'],
        args = training_args
    )
    if len(os.listdir(output_dir)) == 0:
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=True)

def step_train_model_fact_decomp(fact_dataset: pl.DataFrame,
                                 clear_save_dir: bool,
                                 tgt_lang: str,
                                 train_with_peft: bool = False,
                                 use_mt5: bool = False,
                                 **kwargs):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise ValueError("No GPU available.")
    if use_mt5:
        model = MT5ForConditionalGeneration.from_pretrained("google/mt5-large", cache_dir=HF_CACHE_DIR).to(device)
        tokenizer = AutoTokenizer.from_pretrained("google/mt5-large", cache_dir=HF_CACHE_DIR)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", cache_dir=HF_CACHE_DIR).to(device)
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    fact_dataset = fact_dataset.sample(fraction=1.0, shuffle=True)
    hf_dataset = Dataset.from_pandas(fact_dataset.to_pandas()).map(partial(preprocess_fact_decomposition, tokenizer), remove_columns=['prompt', 'facts'])
    max_train_length = max([len(x) for x in hf_dataset['labels']]) # TODO: shouldn't this be labels instead?
    logger.info(f"Max train length: {max_train_length}") 
    hf_dataset = hf_dataset.train_test_split(test_size=0.1)
    lr = 1e-3
    num_epochs = 20
    output_dir = f"{FACT_DECOMP_FLAN_SAVE_DIR}_twp={train_with_peft}_tgt_lang={tgt_lang}_lr={lr}_epochs={num_epochs}_use_mt5={use_mt5}"

    if clear_save_dir:
        shutil.rmtree(output_dir, ignore_errors=True)

    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id, 
        padding=True, 
    )
    training_args = Seq2SeqTrainingArguments(
        report_to="wandb",
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        predict_with_generate=True,
        fp16=False, # Overflows with fp16
        learning_rate=lr,
        num_train_epochs=num_epochs,
        generation_max_length=max_train_length,
        # logging & evaluation strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=25,
        eval_steps=100,
        save_steps = 200,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        metric_for_best_model="rouge2",
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=hf_dataset['train'], 
        compute_metrics=partial(compute_metrics_rouge, tokenizer),
        eval_dataset=hf_dataset['test'],
        args = training_args
    )
    if len(os.listdir(output_dir)) == 0:
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=True)


def step_train_model_fact_decomp_hparam_sweep(fact_dataset: pl.DataFrame,
                                 clear_save_dir: bool,
                                 tgt_lang: str,
                                 train_with_peft: bool = False,
                                 use_mt5: bool = False,
                                 **kwargs):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise ValueError("No GPU available.")
    if use_mt5:
        model = MT5ForConditionalGeneration.from_pretrained("google/mt5-large", cache_dir=HF_CACHE_DIR).to(device)
        tokenizer = AutoTokenizer.from_pretrained("google/mt5-large", cache_dir=HF_CACHE_DIR)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", cache_dir=HF_CACHE_DIR).to(device)
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    def model_init():
        if use_mt5:
            model = MT5ForConditionalGeneration.from_pretrained("google/mt5-large", cache_dir=HF_CACHE_DIR).to(device)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", cache_dir=HF_CACHE_DIR).to(device)
        return model
    sweep_config = {
        'method': 'random'
    }
    # hyperparameters
    parameters_dict = {
        'gradient_accumulation_steps': {
            'values': [2, 4, 8],
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3
        },
        'weight_decay': {
            'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        },
    }
    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project='ru-fact-decomp')

    
    hf_dataset = Dataset.from_pandas(fact_dataset.to_pandas()).map(partial(preprocess_fact_decomposition, tokenizer), remove_columns=['prompt', 'facts'])
    max_train_length = max([len(x) for x in hf_dataset['labels']]) # TODO: shouldn't this be labels instead?
    logger.info(f"Max train length: {max_train_length}") 
    hf_dataset = hf_dataset.train_test_split(test_size=0.1)
    output_dir = f"{FACT_DECOMP_FLAN_SAVE_DIR}_twp={train_with_peft}_tgt_lang={tgt_lang}"

    if clear_save_dir:
        shutil.rmtree(output_dir, ignore_errors=True)

    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id, 
        padding=True, 
    )

    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config
            training_args = Seq2SeqTrainingArguments(
                report_to="wandb",
                output_dir=output_dir,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                predict_with_generate=True,
                fp16=False, # Overflows with fp16
                learning_rate=config.learning_rate,
                num_train_epochs=10,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                weight_decay=config.weight_decay,
                generation_max_length=max_train_length,
                # logging & evaluation strategies
                logging_dir=f"{output_dir}/logs",
                logging_strategy="steps",
                logging_steps=25,
                eval_steps=100,
                save_steps = 200,
                evaluation_strategy="steps",
                save_strategy="steps",
                save_total_limit=2,
                metric_for_best_model="rouge2",
                load_best_model_at_end=True,
            )
            trainer = Seq2SeqTrainer(
                model_init=model_init,
                tokenizer=tokenizer,
                data_collator=data_collator,
                train_dataset=hf_dataset['train'], 
                compute_metrics=partial(compute_metrics_rouge, tokenizer),
                eval_dataset=hf_dataset['test'],
                args = training_args
            )
            trainer.train()
    # if len(os.listdir(output_dir)) == 0:
    #     trainer.train()
    # else:
    #     trainer.train(resume_from_checkpoint=True)
    wandb.agent(sweep_id, train, count=10)

def step_train_info_gap_model(ig1_dataset: pl.DataFrame, 
                     ig2_dataset: pl.DataFrame,
                     clear_save_dir: bool, 
                     lang_pair: str,
                     intersection_label: str,
                     use_mt5: bool,
                     train_with_peft: bool = False,
                     ig1_person_name_label: str = 'person_name',
                     ig2_person_name_label: str = 'person_name',
                     do_sweep=False,
                     **kwargs):
    proportion_tests = [10, 15, 20, 25, 30, 35] 
    for prop in proportion_tests:
        ig1_test_para_indices_dict = ig1_dataset.group_by(ig1_person_name_label).agg(pl.col('paragraph_index').sample( pl.col('paragraph_index').count() // prop, with_replacement=False)).to_dict(as_series=False)
        # convert the the dict so it maps person_name to a list of paragraph indices
        ig1_test_para_indices_dict = {ig1_test_para_indices_dict[ig1_person_name_label][i]: ig1_test_para_indices_dict['paragraph_index'][i] for i in range(len(ig1_test_para_indices_dict[ig1_person_name_label]))}
        # add a column to the en_dataset that indicates whether the paragraph index is in the test set
        partition_ig1_dataset = ig1_dataset.with_columns([
            pl.struct([ig1_person_name_label, 'paragraph_index']).map_elements(
             lambda x: (x['paragraph_index'] in ig1_test_para_indices_dict[x[ig1_person_name_label]])).alias('is_test')
        ])
        print(f"{(partition_ig1_dataset['is_test'].sum() / len(partition_ig1_dataset)):0.2f} of the datapoints are in the En test set.")
    # do the same for the fr_dataset
    for prop in proportion_tests:
        ig2_test_para_indices_dict = ig2_dataset.group_by(ig2_person_name_label).agg(pl.col('paragraph_index').sample( pl.col('paragraph_index').count() // prop, with_replacement=False)).to_dict(as_series=False)
        # convert the the dict so it maps person_name to a list of paragraph indices
        ig2_test_para_indices_dict = {ig2_test_para_indices_dict[ig2_person_name_label][i]: ig2_test_para_indices_dict['paragraph_index'][i] for i in range(len(ig2_test_para_indices_dict[ig2_person_name_label]))}
        # add a column to the en_dataset that indicates whether the paragraph index is in the test set
        partition_ig2_dataset = ig2_dataset.with_columns([
            pl.struct([ig2_person_name_label, 'paragraph_index']).map_elements(
                lambda x: (x['paragraph_index'] in ig2_test_para_indices_dict[x[ig2_person_name_label]])).alias('is_test')
        ])
        print(f"{(partition_ig2_dataset['is_test'].sum() / len(partition_ig2_dataset)):0.2f} of the datapoints are in the Fr test set.")
    def model_init():
        if use_mt5:
            model = MT5ForConditionalGeneration.from_pretrained("google/mt5-large", cache_dir=HF_CACHE_DIR).to(device)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", cache_dir=HF_CACHE_DIR).to(device)
        return model

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise ValueError("No GPU available.")
    
    if not use_mt5:
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", cache_dir=HF_CACHE_DIR).to(device)
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    else:
        # model = MT5ForSequenceClassification.from_pretrained("google/mt5-large", cache_dir=HF_CACHE_DIR).to(device)
        model = MT5ForConditionalGeneration.from_pretrained("google/mt5-large", cache_dir=HF_CACHE_DIR).to(device)
        tokenizer = AutoTokenizer.from_pretrained("google/mt5-large", cache_dir=HF_CACHE_DIR)

    sweep_config = {
        'method': 'random'
    }
    # hyperparameters
    parameters_dict = {
        'gradient_accumulation_steps': {
            'values': [2, 4, 8],
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3
        },
        'weight_decay': {
            'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        },
    }
    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project='ru-infogap-classification_sweep')
    
    # rename the person_name column to 'person_name' for both datasets
    partition_ig1_dataset = partition_ig1_dataset.rename({ig1_person_name_label: 'person_name'})
    partition_ig2_dataset = partition_ig2_dataset.rename({ig2_person_name_label: 'person_name'})

    train_dataset = pl.concat([partition_ig1_dataset.filter(~pl.col('is_test')), partition_ig2_dataset.filter(~pl.col('is_test'))])
    # shuffle the dataset
    train_dataset = train_dataset.sample(fraction=1.0, shuffle=True)
    dev_dataset = pl.concat([partition_ig1_dataset.filter(pl.col('is_test')), partition_ig2_dataset.filter(pl.col('is_test'))])

    preprocess_info_gap_prt = partial(preprocess_info_gap, intersection_label)
    hf_train_dataset = Dataset.from_pandas(train_dataset.to_pandas()).map(partial(preprocess_info_gap_prt, tokenizer), remove_columns=['info_gap_prompt', intersection_label, 'person_name', 'paragraph_index'])
    hf_dev_dataset = Dataset.from_pandas(dev_dataset.to_pandas()).map(partial(preprocess_info_gap_prt, tokenizer), remove_columns=['info_gap_prompt', intersection_label, 'person_name', 'paragraph_index'])
    hf_dataset = DatasetDict({'train': hf_train_dataset, 'test': hf_dev_dataset})
    max_train_length = max([len(x) for x in hf_train_dataset['labels']])
    assert max_train_length == 2 # one for 'yes' or 'no' + 1 for the end token
    # hf_dataset = hf_dataset.train_test_split(test_size=0.1)
    output_dir = f"{CONNOTATION_FLAN_SAVE_DIR}_info_gap_twp={train_with_peft}_lang_pair={lang_pair}_mt5={use_mt5}_do_sweep={do_sweep}"
    # create output directory if it doesn't exist
    if clear_save_dir:
        shutil.rmtree(output_dir, ignore_errors=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    label_pad_token_id = -100

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id, 
        padding=True, 
    )
    training_args = Seq2SeqTrainingArguments(
        report_to="wandb",
        output_dir=output_dir,
        predict_with_generate=True,
        fp16=False, # Overflows with fp16
        learning_rate=1.89e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=5,
        gradient_accumulation_steps=4,
        generation_max_length=max_train_length,
        # logging & evaluation strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=25,
        eval_steps=100,
        save_steps = 200,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        weight_decay=0.4,
        metric_for_best_model="micro-f1",
        load_best_model_at_end=True,
    )
    trainer = Seq2SeqTrainer(
        model = model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=hf_dataset['train'], 
        compute_metrics=partial(compute_metrics_info_gap, tokenizer),
            eval_dataset=hf_dataset['test'],
            args = training_args
        )
    trainer.train()

@click.command()
@click.option('--clear_save_dir', is_flag=True)
@click.option('--train_with_peft', is_flag=True)
def train_connotation_model_pipeline(clear_save_dir, train_with_peft):
    steps = OrderedDict()
    steps['load_dataset_fr'] = SingletonStep(step_load_dataset, {
        'version': '001', 
        'dataset_path': os.path.join(SCRATCH_DIR, 
                                    'ethics-lgbt-data',
                                    'french_connotations_updated_2024_04_09.json'),
        'lang': 'fr'
    })
    steps['load_dataset_en'] = SingletonStep(step_load_dataset, {
        'version': '001', 
        'dataset_path': os.path.join(SCRATCH_DIR, 
                                    'ethics-lgbt-data',
                                    'english_connotations_no_err_2024_04_12.json'),
        'lang': 'en' 
    })
    steps['train_model'] = SingletonStep(step_train_model, {
        'version': '001',
        'fr_dataset': 'load_dataset_fr', 
        'en_dataset': 'load_dataset_en',
        'clear_save_dir': clear_save_dir, 
        'train_with_peft': train_with_peft 
    })
    metadata = conduct(os.path.join(SCRATCH_DIR, 'full_cache'), steps, "finetune_logs")
    print(metadata)

@click.command()
@click.option('--clear_save_dir', is_flag=True)
@click.option('--train_with_peft', is_flag=True)
def train_en_ru_connotation_model_pipeline(clear_save_dir, train_with_peft):
    steps = OrderedDict()
    steps['load_dataset_ru'] = SingletonStep(step_load_dataset, {
        'version': '001', 
        'dataset_path': os.path.join(SCRATCH_DIR, 
                                    'ethics-lgbt-data',
                                    'connotation_df_en_ru_gpt_2024_05_30.json'),
        'lang': 'ru'
    })
    steps['train_model'] = SingletonStep(step_train_model, {
        'version': '001',
        'lang_one_dataset': 'load_dataset_ru',
        'fr_dataset': None,
        'langs': 'ru',
        'clear_save_dir': clear_save_dir, 
        'train_with_peft': train_with_peft,
        'use_mt5': True
    })
    metadata = conduct(os.path.join(SCRATCH_DIR, 'full_cache'), steps, "finetune_logs")
    print(metadata)


@click.command()
@click.option('--clear_save_dir', is_flag=True)
@click.option('--train_with_peft', is_flag=True)
def train_info_gap_model_pipeline(clear_save_dir, train_with_peft):
    assert clear_save_dir
    steps = OrderedDict()
    steps['load_ig_dataset_en'] = SingletonStep(step_load_infogap_dataset, {
        'version': '002',
        'src_lang': 'en',
        'tgt_lang': 'fr',
        'src_dataset_path': os.path.join(SCRATCH_DIR, 'ethics-lgbt-data', 'en_info_gap.json'),
        'tgt_dataset_path': os.path.join(SCRATCH_DIR, 'ethics-lgbt-data', 'fr_info_gap.json')
    })
    steps['load_ig_dataset_fr'] = SingletonStep(step_load_infogap_dataset, {
        'version': '002',
        'src_lang': 'fr',
        'tgt_lang': 'en',
        'src_dataset_path': os.path.join(SCRATCH_DIR, 'ethics-lgbt-data', 'fr_info_gap.json'),
        'tgt_dataset_path': os.path.join(SCRATCH_DIR, 'ethics-lgbt-data', 'en_info_gap.json')
    })
    steps['train_model_info_gap'] = SingletonStep(step_train_info_gap_model, {
        'version': '001',
        'fr_dataset': 'load_ig_dataset_fr', 
        'en_dataset': 'load_ig_dataset_en',
        'clear_save_dir': clear_save_dir, 
        'train_with_peft': train_with_peft 
    })
    metadata = conduct(os.path.join(SCRATCH_DIR, 'full_cache'), steps, "finetune_logs")

@click.command()
@click.option('--clear_save_dir', is_flag=True)
def train_en_ru_info_gap_model_pipeline(clear_save_dir):
    steps = OrderedDict()
    steps['load_ig_dataset_en'] = SingletonStep(step_load_infogap_dataset, {
        'version': '001',
        'src_lang': 'en',
        'tgt_lang': 'ru',
        'src_dataset_path': os.path.join(SCRATCH_DIR, 'ethics-lgbt-data', 'info_gap_en_gpt_05_30.json'),
        'tgt_dataset_path': os.path.join(SCRATCH_DIR, 'ethics-lgbt-data', 'info_gap_ru_gpt_05_30.json'),
        'ig_label': 'gpt4v_intersection_label'
    })
    steps['load_ig_dataset_ru'] = SingletonStep(step_load_infogap_dataset, {
        'version': '002',
        'src_lang': 'ru',
        'tgt_lang': 'en',
        'src_dataset_path': os.path.join(SCRATCH_DIR, 'ethics-lgbt-data', 'info_gap_ru_gpt_05_30.json'),
        'tgt_dataset_path': os.path.join(SCRATCH_DIR, 'ethics-lgbt-data', 'info_gap_en_gpt_05_30.json'),
        'ig_label': 'gpt4v_intersection_label'
    })
    steps['train_model_info_gap'] = SingletonStep(step_train_info_gap_model, {
        'version': '001',
        'ig1_dataset': 'load_ig_dataset_ru', 
        'ig2_dataset': 'load_ig_dataset_en',
        'ig1_person_name_label': 'ru_person_name',
        'ig2_person_name_label': 'person_name',
        'lang_pair': 'ru_en',
        'intersection_label': 'gpt4v_intersection_label',
        'clear_save_dir': clear_save_dir, 
        'use_mt5': True,
        'do_sweep': False
    })
    metadata = conduct(os.path.join(SCRATCH_DIR, 'full_cache'), steps, "finetune_logs")

def step_load_fact_dataset(dataset_path, **kwargs):
    dataset = pl.read_json(dataset_path)
    dataset = dataset.with_columns([
        pl.struct(['paragraph', 'language']).map_elements(
            lambda x: construct_fact_decomp_prompt(x['language'], x['paragraph'])
        ).alias('prompt')
    ]).select(['prompt', 'facts'])
    return dataset

@click.command()
@click.option('--clear_save_dir', is_flag=True)
@click.option('--train_with_peft', is_flag=True)
def train_fact_decomp_model_pipeline(clear_save_dir, train_with_peft):
    steps = OrderedDict()
    steps['load_fact_dataset'] = SingletonStep(step_load_fact_dataset, {
        'version': '002', 
        'dataset_path': os.path.join(SCRATCH_DIR, 
                                    'ethics-lgbt-data',
                                    'fact_dataset_en_fr.json')
    })
    steps['train_model_fact_decomp'] = SingletonStep(step_train_model_fact_decomp, {
        'version': '001',
        'fact_dataset': 'load_fact_dataset', 
        'clear_save_dir': clear_save_dir, 
        'train_with_peft': train_with_peft 
    })
    metadata = conduct(os.path.join(SCRATCH_DIR, 'full_cache'), steps, "finetune_logs")
    print(metadata)

def step_prepare_fact_dataset(complete_dataset_path: str, **kwargs):
    # connotation_df_en_ru_gpt_2024_05_30.json
    frame = pl.read_json(complete_dataset_path)
    all_paragraphs = [] # list of strs
    facts = [] # list of lists
    ru_bio_ids = frame['ru_bio_id'].unique().to_list()
    for ru_bio_id in tqdm(ru_bio_ids):
        person_frame = frame.filter(pl.col('ru_bio_id') == ru_bio_id) 
        ru_bio_content = step_retrieve_prescraped_ru_content_blocks(ru_bio_id)
        ru_paragraphs = list(filter(lambda x: isinstance(x, Paragraph), ru_bio_content))
        ru_paragraph_indices = sorted(person_frame['paragraph_index'].unique().to_list())
        ru_facts = []
        for paragraph_index in ru_paragraph_indices:
            ru_facts.append(
                '\n'.join(person_frame.filter(pl.col('paragraph_index') == paragraph_index)['fact'].to_list())
            )
            all_paragraphs.append(ru_paragraphs[paragraph_index].clean_text)
        for ru_fact_block in ru_facts:
            facts.append(ru_fact_block)
        assert len(ru_paragraph_indices) == len(ru_facts), ipdb.set_trace()
    fact_dataset = pl.DataFrame({
        'paragraph': all_paragraphs,
        'facts': facts
    }).with_columns(pl.lit('ru').alias('language'))\
        .write_json(os.path.join(SCRATCH_DIR, 'ethics-lgbt-data', 'fact_dataset_ru.json'))
    return fact_dataset
    # create a json file that has the fact decomposition (list) 
    # and the paragraph (str) for each person

@click.command()
@click.option('--clear_save_dir', is_flag=True)
@click.option('--do_sweep', is_flag=True)
def train_en_ru_fact_decomp_model_pipeline(clear_save_dir, do_sweep):
    steps = OrderedDict()
    steps['step_prepare_fact_dataset'] = SingletonStep(step_prepare_fact_dataset, {
        'complete_dataset_path': f'{SCRATCH_DIR}/ethics-lgbt-data/info_gap_ru_gpt_05_30.json',
        'version': '002'
    })
    steps['step_load_fact_dataset'] = SingletonStep(step_load_fact_dataset, {
        'version': '003',
        'dataset_path': os.path.join(SCRATCH_DIR, 
                                     'ethics-lgbt-data', 
                                     'fact_dataset_ru.json')
    })
    if do_sweep:
        steps['step_train_model_fact_decomp_sweep'] = SingletonStep(step_train_model_fact_decomp_hparam_sweep, {
            'version': '001',
            'fact_dataset': 'step_load_fact_dataset',
            'clear_save_dir': clear_save_dir, 
            'tgt_lang': 'ru', 
            'use_mt5': True
        })
    else:
        steps['step_train_model_fact_decomp'] = SingletonStep(step_train_model_fact_decomp, {
            'version': '001',
            'fact_dataset': 'step_load_fact_dataset',
            'clear_save_dir': clear_save_dir, 
            'tgt_lang': 'ru', 
            'train_with_peft': False,
            'use_mt5': True
        })
    metadata = conduct(os.path.join(SCRATCH_DIR, 'full_cache'), steps, "finetune_logs")

@click.group()
def main():
    pass

main.add_command(train_connotation_model_pipeline)
main.add_command(train_info_gap_model_pipeline)
main.add_command(train_fact_decomp_model_pipeline)

main.add_command(train_en_ru_fact_decomp_model_pipeline)
main.add_command(train_en_ru_info_gap_model_pipeline)
main.add_command(train_en_ru_connotation_model_pipeline)


if __name__ == '__main__':
    main()