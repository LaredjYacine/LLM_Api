from fastapi import FastAPI
from collections.abc import AsyncIterable,Iterable
from qwen_tokenizer import get_tokenizer
from fastapi.sse import EventSourceResponse ,ServerSentEvent
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(
    base_url='http://localhost:11434/v1/'
, 
   api_key='ollama'
    
)

NomadicModel='nomic-embed-text:latest'
qwen= 'qwen2.5-coder:3b'


class Stream_Output(BaseModel):
    output : str




def streamAi(input:str):
    stream =   client.responses.create(
        model=qwen,
        input=input,   
        stream=True,
 
        
        
        )
    token = 0
    for event in stream:
        if hasattr(event,'delta') and event.delta is not None and getattr(event, "type", None) == "response.output_text.delta":

          tokenizer= get_tokenizer('Qwen/Qwen2.5-Coder-3B-Instruct')
          token += len(tokenizer.encode(event.delta))

          yield event.delta
    yield {'token':token}




def parsedStream(input:str):
    with client.responses.stream(
        model=qwen,
        input=input,
        text_format=Stream_Output
    )as stream:
        for event in stream:
            if getattr(event, "type", None) == "response.output_text.delta":
                yield event.delta




parsedStream('whats your name identify urself')

# app = FastAPI()


# @app.get('/Stream/{input}' ,response_class=EventSourceResponse)
# def streamresponse(input : str ):
#     res =streamAi(input)
    
#     return res  
    

# @app.get('/Parsed/{input}', responses=EventSourceResponse)
# def response(input:str):
#     return parsedStream(input)
