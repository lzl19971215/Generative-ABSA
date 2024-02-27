import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import T5ForConditionalGeneration, T5Tokenizer

# model_name = 'FacebookAI/roberta-large'
model_name = 'google-t5/t5-base'
# T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=f'../bert_models/{os.path.basename(model_name)}')
T5Tokenizer.from_pretrained(model_name)