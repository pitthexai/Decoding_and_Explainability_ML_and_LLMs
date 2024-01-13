from openai import OpenAI

from lib.consts import openai_api_key


class ChatGPT:
    def __init__(self):
        self.api = OpenAI(api_key=openai_api_key)
        self.final_prompt: str
        self.prompts = []

    def add_prompt(self, prompt: dict):
        self.prompts.append(prompt)
        return self

    def ask(self):
        completion = self.api.chat.completions.create(
            model='gpt-3.5-turbo-16k',
            messages=self.prompts,
        )

        return completion.choices[0].message.content
