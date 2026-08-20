from fastapi import FastAPI
from qwen_tokenizer import get_tokenizer

from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/'
,    api_key='ollama'
    
)

NomadicModel='nomic-embed-text:latest'
qwen= 'qwen2.5-coder:3b'


def streamAi():

    stream =   client.responses.create(
        model=qwen,
        input='hey',   
        stream=True
        )
    return stream


                
print('\n')



app = FastAPI()

@app.get('/talk_Ai')
def talk_ai():
    tokenizer= get_tokenizer('Qwen/Qwen2.5-Coder-3B-Instruct')
    stream =  streamAi()
    last_charecter=[]
    return {'output':stream}
#    tokenz =tokenizer.encode(''.join(last_charecter))
