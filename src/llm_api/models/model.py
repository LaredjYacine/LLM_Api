from sqlalchemy.orm import declarative_base,mapped_column,Mapped
from sqlalchemy import Column,BigInteger

from pgvector import Vector
class Base(declarative_base):
    pass


class EmbeddingVector(Base):
    __tablename__='EmbeddingV'
    id : Mapped[int] =mapped_column(type_=BigInteger,primary_key=True) 
    embedding : Mapped[list[float]] =mapped_column( Vector(768))
