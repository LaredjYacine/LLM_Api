from sqlalchemy import create_engine,Engine
from sqlalchemy.orm import sessionmaker
from .Credintials import username,hostname,password,port

def getsyncConnection():
    try :
        engine = create_engine(url=f'postgresql+psycopg2://{username}:{password}@{hostname}:{port}/postgres')
        return engine
    except Exception as e : 
        print('error occured : ', e )




def createSyncSession(engine:Engine):
    try : 
        Session = sessionmaker(bind=engine)
        session = Session()
        return session
    except Exception as e: 
        print('an error occured when creating a Session ', e )
