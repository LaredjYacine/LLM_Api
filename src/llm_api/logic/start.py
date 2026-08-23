from fastapi import FastAPI
from openai import OpenAI
import pandas as pd 
import numpy as np
from ..models.Credintials import qwen,NomicModel
from ..models.Connections import getsyncConnection,create_engine
from ..models.model import EmbeddedItems,Base
from sqlalchemy.orm import Session
from sqlalchemy import insert
client = OpenAI(base_url="http://localhost:11434/v1/",api_key='ollama')



engine = create_engine()
Base.metadata.create_all(engine)
session :Session = getsyncConnection(engine)


df = pd.read_csv('archive/Reviews.csv')

df = df[:101]
df['Combined'] = df['Text'] + ' ' + df['Summary']


df.drop(combined=['Text','Summary','id','ProductId','UserId','HelpfulnessNumerator','HelpfulnessDenominator','Score','Time'])

for index , row in df.iterrows() :
    batch=[]
    Combined= row['Combined']
    ProfileName= row['ProfileName']
    vector =  client.embeddings.create(input=row['Combined'], model=NomicModel).data[0].embedding     
    row ={
        'ProfileName': ProfileName,
        'Combined':Combined,
        'Embedding':vector,
    }
    batch.append(row)
    if len(batch)>100:
        session.execute(insert(EmbeddedItems), batch)
        session.commit()
    if batch :
        session.execute(insert(EmbeddedItems), batch)
        session.commit()
        

     
        

    
    
    





# def cosineSimilarities(A,B):
#     product = np.dot(np.array(A),np.array(B))
#     MagnitudeA = np.linalg.norm(np.array(A))
#     MagnitudeB = np.linalg.norm(np.array(B))
#     dotmag=MagnitudeA * MagnitudeB
#     CosineSimilarity = np.divide(product,dotmag)
#     return CosineSimilarity 

# def getEmbedding(text, NomicModel):
#     text = text.replace('\n', ' ')
#     return client.embeddings.create(input = text,model=NomicModel).data[0].embedding

# df=pd.read_csv('archive/Embeddedoutput') 
# df=df.drop(columns=['Text','Summary'])
# df['ada_embedding']=df['ada_embedding'].apply(eval).apply(np.array)



# def search_embedding(df:pd.DataFrame, query,pprint=True):
#     embedding = getEmbedding(query,NomicModel)
#     df['similarities'] = df['ada_embedding'].apply(
#         lambda x: cosineSimilarities(x,embedding)
#     )
#     res = df.sort_values('similarities', ascending=False)
#     return res


# def i_guess_this_is_Rag():
#     query = str(input('hello Welcome what do you want to ask  ? \n'))
#     res = search_embedding(df,query)
#     res = res.head(5)
#     res['Rag_column']= 'the Review: '+ res['Combined'] + ' \n Profile Name: ' +res['ProfileName']
#     column  =res['Rag_column'].str.cat(sep='\n---\n')
        

#     responses = client.responses.create(
#         model=qwen,
#         input=[{

#             'role':'system',
# 'content': (
#     "You are a closed-domain recommendation bot. Your knowledge base consists EXCLUSIVELY of the text provided below. "
#     "You have no outside world knowledge. "
#     "Instructions:\n"
#     "1. Answer user questions using ONLY the provided context.\n"
#     "2. If the user asks for translations, general facts, math, or anything not explicitly found in the context, you must output: 'I don't know.'\n"
#     "3. Never use outside knowledge, even for simple questions like translations or greetings.\n\n"
#     "Output Example: - Product/Review: [Details] (Recommended by [Profile Name])\n\n"
#     f"--- BEGIN CONTEXT ---\n{column}\n--- END CONTEXT ---"
# )

#         },
#             {
#                 'role':'user',
#                 'content':query
#             }
            

#             ]
#         ,
#         stream=True
#     )

#     for response in responses:
#         if hasattr(response,'delta') and response.delta is not None :
#             print(response.delta, end='',flush=True)

# i_guess_this_is_Rag()
