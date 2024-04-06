import gc
import torch
import warnings

warnings.filterwarnings("ignore") # good for demo, not good idea for production :)

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "speakleash/Bielik-7B-Instruct-v0.1-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=False,
    revision="main"
)

pipe = pipeline(model=model, tokenizer=tokenizer, task='text-generation')
gptq=True