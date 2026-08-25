from .Connections import getsyncConnection,createSyncSession
from sqlalchemy.orm import Session 
from sqlalchemy import select
from .model import EmbeddedItems
import pandas as pd
import numpy as np 
import bm25s as bm
import Stemmer

engine = getsyncConnection()
stemmer = Stemmer.Stemmer('english')



def Extract():
    session : Session= createSyncSession(engine)
    try :
        Rows = session.execute(select(EmbeddedItems.id, EmbeddedItems.Combined)).all()
        return [{"id": str(row.id), "text": str(row.Combined)} for row in Rows]
    finally: 
        session.close()



def Transform(data):
    if not  data: 
        raise ValueError('no data found to index')

    text = [item['text'] for item in data]
    Bmdata= bm.tokenize(text,stemmer=stemmer,stopwords='en')
    retriever = bm.BM25()
    retriever.index(Bmdata)
    retriever.save('bm25Retriever', corpus=data)

def load():
    retriever_corpus =bm.BM25.load('bm25Retriever', load_corpus=True)
    return retriever_corpus



    
Rows = Extract()
Transform(Rows)
retriever = load()


