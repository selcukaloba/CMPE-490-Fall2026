import os
from openai import OpenAI

api_key = os.environ.get("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model = "gemini-3.8-flash",
    messages = [
        {"role": "user", "content": "What is the capital city of France?"}      
    ] 
)
print(response.choices[0].message.content)