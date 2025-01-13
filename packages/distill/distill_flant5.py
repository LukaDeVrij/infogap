from transformers import T5Tokenizer, T5ForConditionalGeneration
from dotenv import load_dotenv
from peft import LoraConfig, TaskType

peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
# import python-dotenv
import os


tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl", cache_dir=os.getenv("CACHE_DIR"))
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", cache_dir=os.getenv("CACHE_DIR"))

def load_dataset() -> : # TODO: test out on the connotation dataset to see if it helps.
    # a map style dataset.
    pass

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)