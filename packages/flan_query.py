import ipdb
from typing import List
from transformers import MT5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, BitsAndBytesConfig


from packages.gpt_query import construct_fact_decomp_prompt, construct_fact_intersection_prompt
# from packages.steps.caa_steps import construct_prompt_2_en, construct_prompt_2_fr

def generate_facts_flan(model_cache_path: str, lang_code: str):
    flan_model = AutoModelForSeq2SeqLM.from_pretrained(model_cache_path).to('cuda')
    flan_tokenizer = AutoTokenizer.from_pretrained(model_cache_path)
    
    def generate_facts_flan(paragraphs: List[str]): # containing around 1-3 paragraphs.
        prompts = [construct_fact_decomp_prompt(lang_code, paragraph) for paragraph in paragraphs]
        inputs = flan_tokenizer(prompts, padding=True, return_tensors='pt').to('cuda')
        outputs = flan_model.generate(**inputs, max_length=1024)
        return flan_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return generate_facts_flan

    def _generate_facts_flan(paragraph: str):
        prompt = construct_fact_decomp_prompt(lang_code, paragraph)
        inputs = flan_tokenizer(prompt, return_tensors='pt').to('cuda')
        outputs = flan_model.generate(**inputs, max_length=1024)
        return flan_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return _generate_facts_flan

def generate_facts_mt5(model_cache_path: str, lang_code: str):
    mt5_model = MT5ForConditionalGeneration.from_pretrained(model_cache_path).to('cuda')
    mt5_tokenizer = AutoTokenizer.from_pretrained(model_cache_path)

    def generate_facts_mt5(paragraphs: List[str]): # containing around 1-3 paragraphs.
        prompts = [construct_fact_decomp_prompt(lang_code, paragraph) for paragraph in paragraphs]
        inputs = mt5_tokenizer(prompts, padding=True, return_tensors='pt').to('cuda')
        outputs = mt5_model.generate(**inputs, max_length=1024)
        return mt5_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return generate_facts_mt5

    def _generate_facts_mt5(paragraph: str):
        prompt = construct_fact_decomp_prompt(lang_code, paragraph)
        inputs = mt5_tokenizer(prompt, return_tensors='pt').to('cuda')
        outputs = mt5_model.generate(**inputs, max_length=1024)
        return mt5_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return _generate_facts_mt5

def ask_flan_about_fact_intersection(model_cache_path):
    flan_model = AutoModelForSeq2SeqLM.from_pretrained(model_cache_path).to('cuda')
    flan_tokenizer = AutoTokenizer.from_pretrained(model_cache_path)

    def _generate_prompt(src_lang, tgt_lang_code, src_fact_context: List[str], tgt_fact_context: List[List[str]], person_name: str):
        prompt = construct_fact_intersection_prompt(src_lang, tgt_lang_code, src_fact_context, tgt_fact_context, person_name)
        return prompt

    def _facts_intersect(prompts: List[str]):
        # assert that the type of prompts is a list of strings
        assert isinstance(prompts, list)
        assert all(isinstance(prompt, str) for prompt in prompts)
        inputs = flan_tokenizer(prompts, padding=True, return_tensors='pt').to('cuda')
        outputs = flan_model.generate(**inputs)
        return flan_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return _generate_prompt, _facts_intersect

def ask_mt5_about_fact_intersection(model_cache_path):
    mt5_model = MT5ForConditionalGeneration.from_pretrained(model_cache_path).to('cuda')
    mt5_tokenizer = AutoTokenizer.from_pretrained(model_cache_path)

    def _generate_prompt(src_lang, tgt_lang_code, src_fact_context: List[str], tgt_fact_context: List[List[str]], person_name: str):
        prompt = construct_fact_intersection_prompt(src_lang, tgt_lang_code, src_fact_context, tgt_fact_context, person_name)
        return prompt
    
    def _facts_intersect(prompts: List[str]):
        # assert that the type of prompts is a list of strings
        assert isinstance(prompts, list)
        assert all(isinstance(prompt, str) for prompt in prompts)
        inputs = mt5_tokenizer(prompts, padding=True, return_tensors='pt').to('cuda')
        outputs = mt5_model.generate(**inputs)
        return mt5_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return _generate_prompt, _facts_intersect

def ask_flan_about_connotation(model_cache_path, construct_prompt_fn):
    flan_model = AutoModelForSeq2SeqLM.from_pretrained(model_cache_path).to('cuda')
    flan_tokenizer = AutoTokenizer.from_pretrained(model_cache_path)

    def _generate_connotation_prompt(content, person_name, pronoun, lang_code):
        if lang_code == 'en':
            prompt = construct_prompt_fn(content, person_name, pronoun)
        elif lang_code == 'fr':
            prompt = construct_prompt_fn(content, person_name, pronoun)
        elif lang_code == 'ru':
            prompt = construct_prompt_fn(content, person_name, pronoun)
        else:
            raise ValueError(f"Invalid language code: {lang_code}")
        return prompt

    def _predict_connotation(prompts: List[str]):
        inputs = flan_tokenizer(prompts, padding=True, return_tensors='pt').to('cuda')
        outputs = flan_model.generate(**inputs, max_length=1024)
        return flan_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return _generate_connotation_prompt, _predict_connotation

def ask_mt5_about_connotation(model_cache_path, construct_prompt_fn):
    mt5_model = MT5ForConditionalGeneration.from_pretrained(model_cache_path).to('cuda')
    mt5_tokenizer = AutoTokenizer.from_pretrained(model_cache_path)

    def _generate_connotation_prompt(content, person_name, pronoun, lang_code):
        if lang_code == 'en':
            prompt = construct_prompt_fn(content, person_name, pronoun)
        elif lang_code == 'fr':
            prompt = construct_prompt_fn(content, person_name, pronoun)
        elif lang_code == 'ru':
            prompt = construct_prompt_fn(content, person_name, pronoun)
        else:
            raise ValueError(f"Invalid language code: {lang_code}")
        return prompt

    def _predict_connotation(prompts: List[str]):
        inputs = mt5_tokenizer(prompts, padding=True, return_tensors='pt').to('cuda')
        outputs = mt5_model.generate(**inputs, max_length=1024)
        return mt5_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return _generate_connotation_prompt, _predict_connotation
