from fastapi import FastAPI
from openai import OpenAI
import pandas as pd 
import numpy as np
from ..models.Credintials import NomicModel,qwen
from ..models.Connections import createSyncSession,getsyncConnection
from ..models.model import EmbeddedItems,Base
from sqlalchemy.orm import Session
from sqlalchemy import insert,select
client = OpenAI(base_url="http://localhost:11434/v1/",api_key='ollama')



engine = getsyncConnection()
Base.metadata.create_all(engine)
session :Session = createSyncSession(engine)

def search_embedding(query:str):
    results=[]
    embedded =np.array( client.embeddings.create(model= NomicModel, input=query ).data[0].embedding)
    result = session.scalars(select(EmbeddedItems).order_by(EmbeddedItems.Embedding.cosine_distance(embedded)).limit(5)).all()
    for item in result:
        format_chunk = f"the product review : {item.Combined} | Reviwer : {item.ProfileName}\n"
        results.append(format_chunk)
    chat_bot_input = '\n'.join(results)
    return chat_bot_input
        

def i_guess_this_is_Rag():
    while True : 
        query = str(input('\n hello Welcome what do you want to ask  ? \n'))
        if query == 'exit':
            break
        res = search_embedding(query)
            

        responses = client.responses.create(
            model=qwen,
            input=[{

                'role':'system',
    'content': (
        "You are a closed-domain recommendation bot. Your knowledge base consists EXCLUSIVELY of the text provided below. "
        "You have no outside world knowledge. "
        "Instructions:\n"
        "1. Answer user questions using ONLY the provided context.\n"
        "2. If the user asks for translations, general facts, math, or anything not explicitly found in the context, you must output: 'I don't know.'\n"
        "3. Never use outside knowledge, even for simple questions like translations or greetings.\n\n"
        "Output Example: - Product/Review: [Details] (Recommended by [Profile Name])\n\n"
        f"--- BEGIN CONTEXT ---\n{res}\n--- END CONTEXT ---"
    )

            },
                {
                    'role':'user',
                    'content':query
                }
                

                ]
            ,
            stream=True
        )

        for response in responses:
            if hasattr(response,'delta') and response.delta is not None :
                print(response.delta, end='',flush=True)


i_guess_this_is_Rag()
