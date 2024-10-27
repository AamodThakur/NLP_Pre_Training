from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math

# Load model and tokenizer
model_name = "my model path"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = math.exp(loss.item())

    return perplexity

text = "I want to test it"
perplexity = calculate_perplexity(text)
print(f"Perplexity: {perplexity}")
