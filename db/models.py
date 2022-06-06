from ast import In
from typing import Counter
from numpy import integer
from sklearn.compose import ColumnTransformer
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime
from sqlalchemy.orm import relationship

from database import Base

class Image(Base):
    __tablename__ = "images"

    image_id = Column(Integer, primary_key=True, index=True)
    lesion_id = Column(String, index=True)
    diagnosis = Column(String, index=True)
    # Type of confirmation for ground truth (histopathology, consensus...)
    diagnosis_type = Column(String, index=True)
    age = Column(Integer, index=True)
    sex = Column(String, index=True)
    localisation = Column(String, index=True)

    file_path = Column(String, index=True)

    # Latent space coordinates
    l0_coordinate = Column(Integer, index=True)
    l1_coordinate = Column(Integer, index=True)
    l2_coordinate = Column(Integer, index=True)
    l3_coordinate = Column(Integer, index=True)
    l4_coordinate = Column(Integer, index=True)
    l5_coordinate = Column(Integer, index=True)
    l6_coordinate = Column(Integer, index=True)
    l7_coordinate = Column(Integer, index=True)
    l8_coordinate = Column(Integer, index=True)
    l9_coordinate = Column(Integer, index=True)



