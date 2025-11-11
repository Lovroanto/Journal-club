import ollama

# List available models (llama3.1 should appear here)
models = ollama.list()
print(models)