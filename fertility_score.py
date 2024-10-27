from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer_path", type=str, default="")
parser.add_argument("--data_path", type=str, default="")
parser.add_argument("--column_name", type=str, default="")
args = parser.parse_args()


if args.tokenizer_path == "":
    raiseError("Please provide tokenizer path as a flag.")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

if args.data_path == "" or args.column_name == "":
    raiseError("Data or column name missing, not give to the code.")
    
df = pd.read_csv(args.data_path)
df = df[args.column_name].tolist()
    
## Get word count in document
df_word_count = []
for index in range(len(df)):
    df_word_count.append(len(word_tokenize(df[index])))
    
np_word_count = np.array(df_word_count)

## Tokenizer Document
token_list = tokenizer(df)["input_ids"]
## Get token count in the document
token_count = []
for index in range(len(token_list)):
    token_count.append(len(token_list[index]))
np_token_count = np.array(token_count)

## Calculate fertility score
f_score = np.mean(np_token_count/np_word_count)


with open("fertility_score.txt", "w") as file:
    file.write(str(f_score))
