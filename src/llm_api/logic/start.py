from fastapi import FastAPI
import os
import dotenv
from openai import OpenAI
import pandas as pd 

import numpy as np
dotenv.load_dotenv()

client = OpenAI(base_url="http://localhost:11434/v1/",api_key='ollama')
NomicModel='nomic-embed-text'

def getEmbedding(text, NomicModel):
    text = text.replace('\n', ' ')
    return client.embeddings.create(input = text,model=NomicModel).data[0].embedding

df = pd.read_csv('archive/Reviews.csv')
df['combined']= df['Summary'] + '' + df['Text']

test= df['combined'].head(5)

df['ada_embedding']= test.apply(
    lambda x : getEmbedding(x,NomicModel)
)

df.to_csv("archive/embedded_1k_reviews.csv", index=False)
df = pd.read_csv("archive/embedded_1k_reviews.csv")
df["ada_embedding"] = df['ada_embedding'].apply(eval).apply(np.array)

# response = client.embeddings.create(
#     model="nomic-embed-text",
#     input='say hi'
# )

#print(response.data[0].embedding)

 
#app = FastAPI()

#@app.get('/chatbot')
