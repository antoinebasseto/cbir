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
    patient_id = Column(Integer, index = True)
    #pictures = relationship("Picture", back_populates="patient")

class Picture(Base):
    __tablename__ = "pictures"

    id = Column(Integer, primary_key = True, index = True)

    picture_id = Column(String, index = True)
    file_path = Column(String, index = True)
    patient_id = Column(Integer, ForeignKey("patients.patient_id"))
    #stub
    similarity = Column(Integer, index = True)

    #sickness
    Abnormal = Column(Boolean, index = True)
    Atelectasis = Column(Boolean, index = True)
    Cardiomegaly = Column(Boolean, index = True)
    Effusion = Column(Boolean, index = True)
    Infiltration = Column(Boolean, index = True)
    Mass = Column(Boolean, index = True)
    Nodule = Column(Boolean, index = True)

    Pneumonia = Column(Boolean, index = True)
    Pneumothorax = Column(Boolean, index = True)
    Consolidation = Column(Boolean, index = True)
    Edema = Column(Boolean, index = True)
    Emphysema = Column(Boolean, index = True)
    Fibrosis =  Column(Boolean, index = True)
    PleuralThickening = Column(Boolean, index = True)
    Hernia = Column(Boolean, index = True)
    Other = Column(Boolean, index = True)


