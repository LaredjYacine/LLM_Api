import bm25s
from openai import OpenAI
import dotenv
import os
import cohere
import Stemmer
from ..models.Connections import createSyncSession, getsyncConnection
from ..models.Credintials import NomicModel
from ..models.model import EmbeddedItems
from sqlalchemy import select 
import numpy as np 



engine = getsyncConnection()
session = createSyncSession(engine=engine)
dotenv.load_dotenv()
cohere_api = os.getenv('cohere')
co = cohere.ClientV2(api_key=cohere_api)
stemmer = Stemmer.Stemmer('english')
client = OpenAI(base_url="http://localhost:11434/v1/",api_key='ollama')

def bm25_Rag(query :str):
    if query is None: 
        raise ValueError('query cannot be empty')
    Hybrid_Data=[]
    try:
        response_corpus= bm25s.BM25.load('bm25Retriever',load_corpus=True)
        corpus = response_corpus.corpus
        query_tokens = bm25s.tokenize(query,stemmer=stemmer,stopwords='en')
        bm , _ = response_corpus.retrieve(query_tokens=query_tokens,corpus=corpus, k=10)
        for docs in bm[0] : 
            Hybrid_Data.append(docs['text'])
        query_embeddings =np.array( client.embeddings.create(
            model=NomicModel,
            input = query
        ).data[0].embedding
        )
        responses = session.scalars(select(EmbeddedItems).order_by(EmbeddedItems.Embedding.cosine_distance(query_embeddings)).limit(10)).all()
        for  response in responses : 
            Hybrid_Data.append(response.Combined)

        return Hybrid_Data
    except Exception as e : 
        print('error occured at function bm25_rag', e)
    finally:
        session.close()
    

query  = str(input('put ur Input here : '))
hybrid_data = bm25_Rag(query)
creme_data=[]
if hybrid_data: 

    response = co.rerank(
        model='rerank-v4.0-pro',
        query=query,
        documents=hybrid_data,
        top_n=5
    )

    for res in response.results[:5] :
        id = res.index
        creme_data.append(hybrid_data[id])


print(creme_data)














