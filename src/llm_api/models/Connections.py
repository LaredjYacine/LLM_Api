from sqlalchemy import create_engine,Engine
from sqlalchemy.orm import sessionmaker,Session
from .Credintials import username,hostname,password,port

url = f"postgresql+psycopg2://{username}:{password}@{hostname}:{port}/postgres"

def getsyncConnection():
    try :
        engine = create_engine(url=url)
        return engine
    except Exception as e : 
        print('error occured : ', e )




def createSyncSession(engine:Engine) -> Session:
    SessionMaker= sessionmaker(bind=engine)
    return SessionMaker()
