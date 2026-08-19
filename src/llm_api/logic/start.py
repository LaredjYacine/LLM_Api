from fastapi import FastAPI
import os
import dotenv
from openai import OpenAI

dotenv.load_dotenv()


client = OpenAI(base_url="http://localhost:11434/v1",api_key='ollama')
response = client.responses.create(
    model="qwen2.5-coder:3b",
    input='say hi'
)

print(response.output_text)

 
#app = FastAPI()

#@app.get('/chatbot')
