from __future__ import annotations
from email.policy import strict

from typing import List, Optional, TypedDict

from pydantic import BaseModel

class PictureBase(BaseModel):
    picture_id: str

class PictureCreate(PictureBase):
    file_path: str

class Picture(PictureBase):
    id: int

    patient_id: int

    # stub
    similarity : int

    # sickness
    Abnormal : bool
    Atelectasis : bool
    Cardiomegaly : bool
    Effusion : bool
    Infiltration : bool
    Mass : bool
    Nodule : bool
    Pneumonia : bool
    Pneumothorax : bool
    Consolidation : bool
    Edema : bool
    Emphysema : bool
    Fibrosis : bool
    Pleural : bool
    Thickening : bool
    Hernia : bool
    Other : bool

    class Config:
        orm_mode = True

class PatientBase(BaseModel):
    patient_id: str

class PatientCreate(PatientBase):
    age: int
    gender: str

class Patient(PatientBase):
    #pictures: List[Picture] = []
    id : int
    age: int
    gender: str
    class Config:
        orm_mode = True