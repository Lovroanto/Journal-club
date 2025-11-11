import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("EleutherAI/llama-3.1")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llama-3.1")

# Now you can use the model for text generation or other tasks
input_text = "Hello, I am a conversational AI..."
output = model.generate(input_text)
print(output)