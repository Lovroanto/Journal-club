import ollama

class LocalAgent:
    def __init__(self, model_name="llama3.1", system_prompt="You are a helpful research assistant."):
        self.model = model_name
        self.system_prompt = system_prompt

    def run(self, user_prompt):
        response = ollama.chat(
            model=self.model,
            messages=[
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
        )
        return response['message']['content']