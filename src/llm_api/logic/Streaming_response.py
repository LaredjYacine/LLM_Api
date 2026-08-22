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

def few_shot(userinput:str):
    res = client.responses.parse(
        model=qwen,
        input=[
            {
                'role':'system',
                'content':'Act as a classifier and Classify the users input either as Positive or Negative'
            }
            ,
            {
                'role':'assistant',
                'content':"""(The food is amazing) This is a Positive     , (The food tastes bland) This is negative
                 (The food was absolutely delicious.) This is Positive
(The food tasted terrible.) This is Negative
(I loved the service here.) This is Positive
(The waiter was extremely rude.) This is Negative
(This movie was fantastic.) This is Positive
(The movie was boring and predictable.) This is Negative
(I had an amazing experience.) This is Positive
(I had a horrible experience.) This is Negative
(The product works perfectly.) This is Positive
(The product stopped working after one day.) This is Negative
(The hotel room was beautiful and clean.) This is Positive
(The hotel room was dirty and uncomfortable.) This is Negative
(The customer support was very helpful.) This is Positive
(Customer support completely ignored my problem.) This is Negative
(This restaurant exceeded my expectations.) This is Positive
(I would never eat at this restaurant again.) This is Negative
(The laptop is fast and reliable.) This is Positive
(The laptop is slow and constantly crashes.) This is Negative
(I really enjoyed this book.) This is Positive
(This book was a complete waste of time.) This is Negative
(The delivery arrived earlier than expected.) This is Positive
(My delivery arrived three days late.) This is Negative
(The staff were friendly and welcoming.) This is Positive
(The staff were unfriendly and unprofessional.) This is Negative
(The new update made the application much better.) This is Positive
(The latest update broke several important features.) This is Negative
(I am very happy with my purchase.) This is Positive
(I regret buying this product.) This is Negative
(The presentation was clear and informative.) This is Positive
(The presentation was confusing and poorly organized.) This is Negative
(The coffee was fresh and flavorful.) This is Positive
(The coffee was cold and tasteless.) This is Negative
(The website is easy to navigate.) This is Positive
(The website is confusing and difficult to use.) This is Negative
(The camera takes excellent pictures.) This is Positive
(The camera produces terrible images in low light.) This is Negative
(The concert was incredible.) This is Positive
(The concert was disappointing.) This is Negative
(I strongly recommend this service.) This is Positive
(I would not recommend this service to anyone.) This is Negative
(Everything went smoothly.) This is Positive
(Everything went completely wrong.) This is Negative
(The quality is better than I expected.) This is Positive
(The quality is much worse than I expected.) This is Negative
(This is one of the best purchases I've made.) This is Positive
(This is one of the worst purchases I've made.) This is Negative
(The experience was pleasant from beginning to end.) This is Positive
(The experience was frustrating from beginning to end.) This is Negative
(I am impressed with the results.) This is Positive
(I am disappointed with the results.) This is Negative
  """
                
            }
            ,
            {
                'role':'user',
                'content':userinput
            }   
        ],
        text_format=Stream_Output
    )
    return res

def zero_shot(userinput:str):
    res = client.responses.parse(
        model=qwen,
        input=[
            {
                'role':'system',
                'content':'Act as a classifier and Classify the users input either as Positive or Negative'
            }
            
  
            ,
            {
                'role':'user',
                'content':userinput
            }   
        ],
        text_format=Stream_Output
    )
    return res

def chainofThought(userinput:str):
    res = client.responses.parse(
        model=qwen,
        input=[
            {
                'role':'system',
                'content':'analyse the users input then show your steps that lead you to decide wether it was negative or positive '
            }
            
  
            ,
            {
                'role':'user',
                'content':userinput
            }   
        ],
        text_format=Stream_Output
    )
    return res


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
            if getattr(event, "type", None) == "response.output_text.delta" and event.delta is not None:
                
                yield event.delta




app = FastAPI()


@app.get('/Stream/{input}' ,response_class=EventSourceResponse)
def streamresponse(input : str ):
    res =streamAi(input)
    
    return res  
    


@app.get('/fewShot/{input}')
def fewshot(input:str):
    res = few_shot(input)
    return res.output_parsed


@app.get('/zeroShot/{input}')
def zeroshot(input:str):
    res = zero_shot(input)
    return res.output_parsed
@app.get('/chainofthought/{input}')
def Cot(input:str):
    res = chainofThought(input)
    return res.output_parsed