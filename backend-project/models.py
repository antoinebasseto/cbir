from ast import In
from typing import Counter
from numpy import integer
from sklearn.compose import ColumnTransformer
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime
from sqlalchemy.orm import relationship

from database import Base

class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key = True, index=True)
    name = Column(String, index = True)
    #pictures = relationship("Picture", back_populates="patient")

class Picture(Base):
    __tablename__ = "pictures"

    id = Column(Integer, primary_key = True, index = True)
    title = Column(String, index = True)
    file_path = Column(String, index = True)
    #patient_id = Column(Integer, ForeignKey("patients.id"))
    
    datetime = Column(DateTime, index = True)
    #stub
    similarity = Column(Integer, index = True)

    #patient = relationship("Patient", back_populates="pictures")

