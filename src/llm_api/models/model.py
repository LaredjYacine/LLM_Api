from sqlalchemy.orm import DeclarativeBase,mapped_column,Mapped
from sqlalchemy import Column,BigInteger,String

from pgvector import Vector
class Base(DeclarativeBase):
    pass



class EmbeddedItems(Base):
    __tablename__ ='EmbeddTable'
    ProfileName=Mapped[str]= mapped_column(String,nullable=False)
    Combined = Mapped[str]=mapped_column(String , nullable=False)
    Embedding : Mapped[list[float]] =mapped_column( Vector(768))
    