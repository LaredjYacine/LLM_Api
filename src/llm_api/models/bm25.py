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
session : Session= createSyncSession(engine)



def Extract():
    Rows = session.scalars(select(EmbeddedItems.Combined)).all()
    return Rows



def Transform(Rows):
    data =[]
    for Row in Rows :
        data.append(Row)
     
    Bmdata= bm.tokenize(data,stemmer=stemmer,stopwords='en')
    retriever = bm.BM25()
    retriever.index(Bmdata)
    retriever.save('bm25Retriever', corpus=data)




def load():
    retriever_corpus =bm.BM25.load('bm25Retriever', load_corpus=True)
    return retriever_corpus



    
Rows = Extract()
Transform(Rows)
retriever = load()



