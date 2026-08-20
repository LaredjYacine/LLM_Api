from fastapi import FastAPI
import os
import dotenv
from openai import OpenAI
import pandas as pd 

import numpy as np
dotenv.load_dotenv()

client = OpenAI(base_url="http://localhost:11434/v1/",api_key='ollama')
NomicModel='nomic-embed-text'


def cosineSimilarities(A,B):
    product = np.dot(np.array(A),np.array(B))
    MagnitudeA = np.linalg.norm(np.array(A))
    MagnitudeB = np.linalg.norm(np.array(B))
    dotmag=MagnitudeA * MagnitudeB
    CosineSimilarity = np.divide(product,dotmag)
    return CosineSimilarity 

def getEmbedding(text, NomicModel):
    text = text.replace('\n', ' ')
    return client.embeddings.create(input = text,model=NomicModel).data[0].embedding

df=pd.read_csv('archive/Embeddedoutput')

df['ada_embedding']=df['ada_embedding'].apply(eval).apply(np.array)



def search_embedding(df:pd.DataFrame, query,pprint=True):
    embedding = getEmbedding(query,NomicModel)
    df['similarities'] = df['ada_embedding'].apply(
        lambda x: cosineSimilarities(x,embedding)
    )
    res = df.sort_values('similarities', ascending=False)
    return res



res = search_embedding(df,' dog')
print(res)

# response = client.embeddings.create(
#     model="nomic-embed-text",
#     input='say hi'
# )

#print(response.data[0].embedding)

 
#app = FastAPI()

#@app.get('/chatbot')
