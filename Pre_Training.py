# -*- coding: utf-8 -*-
"""NLP_TA.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17j5Zz_jwZ6bKE-RQU5A6Dd3-TqBXxYUW
"""

from transformers import AutoModelForCausalLM, GemmaConfig, AutoTokenizer, AutoModel, MistralConfig, MistralModel, MistralForCausalLM, LlamaConfig, LlamaForCausalLM
import torch
import torch.nn as nn
import torch.nn.init as init
import json
import pickle
import pandas as pd

token_path = "tokenizer"
tokenizer = AutoTokenizer.from_pretrained("token")

len(tokenizer.vocab)

config = LlamaConfig(hidden_size=256,
                     vocab_size=len(tokenizer.vocab),
                     num_attention_heads=4,
                     num_key_value_heads=2,
                     num_hidden_layers=8,
                     intermediate_size=688)
config

model_mis = LlamaForCausalLM(config)

total_param=0
for i,j in model_mis.named_parameters():
    total_param += j.numel()
print(total_param/(10**6))

!pip install datasets -q

from datasets import Dataset, DatasetDict
from datasets import load_dataset

dataset = load_dataset("databricks/databricks-dolly-15k")
print(dataset)

data_as_dicts = [{'text': item} for item in dataset["train"]["instruction"]]

dataset = Dataset.from_list(data_as_dicts)
dataset

split_dataset = dataset.train_test_split(test_size=0.2)  # Adjust test_size as needed

train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

train_dataset

!pip install trl -q

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    output_dir="./bert",
    max_seq_length = 512,
    overwrite_output_dir=True,
    dataset_text_field="text",
    num_train_epochs=10,
    bf16=True,
    do_train=True,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
)

trainer = SFTTrainer(
    model=model_mis,
    args=training_args,
    train_dataset=train_dataset,
    #eval_dataset=eval_dataset,
    #max_length=512,
    tokenizer=tokenizer,
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

tokenizer.pad_token = tokenizer.eos_token

trainer.train()

