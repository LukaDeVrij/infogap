from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .constants import HF_CACHE_DIR

model_name = 'google/flan-t5-small'

# https://www.dropbox.com/scl/fi/tsalsi342eha757foeu6h/Instructdial-stratify.zip?rlkey=6xic2tel8vljdn5251o5zcthf&dl=0
class FlanT5():
    # create a static class variable so that the flan model is only loaded once
    _cached_model = None
    _cached_tokenizer = None

    def __init__(self, variant: str) -> None:
        self.variant = variant

    @property
    def tokenizer(self):
        if self._cached_tokenizer is None:
            self._cached_tokenizer = AutoTokenizer.from_pretrained(f"google/{self.variant}", cache_dir= HF_CACHE_DIR)
        return self._cached_tokenizer

    @property
    def model(self):
        if self._cached_model is None:
            self._cached_model = AutoModelForSeq2SeqLM.from_pretrained(f"google/{self.variant}", cache_dir = HF_CACHE_DIR)
        return self._cached_model

def prompt_model(model, tokenizer, prompt, generation_hyperparams):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], do_sample=True, top_p=generation_hyperparams['nucleus_size'], top_k=0, 
                       max_length=200, repetition_penalty=1.2, temperature=generation_hyperparams['temperature'],
                       decoder_start_token_id=model.config.decoder_start_token_id, num_return_sequences=generation_hyperparams['num_generations'],
                       remove_invalid_values=True)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]