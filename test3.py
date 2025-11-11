import ollama

response = ollama.chat(
    model='llama3.1',  # name of your local model
    messages=[
        {'role': 'system', 'content': 'You are a helpful research assistant.'},
        {'role': 'user', 'content': 'Summarize the abstract of this neuroscience paper.'}
    ]
)

print(response['message']['content'])