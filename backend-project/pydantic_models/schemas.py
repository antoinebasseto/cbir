from __future__ import annotations
from email.policy import strict

from typing import List, Optional, TypedDict

from pydantic import BaseModel

class PictureBase(BaseModel):
    title: str

class PictureCreate(PictureBase):
    file_path: str

class Picture(PictureBase):
    id: int
    patient_id: Optional[int] = None
    file_path: str
    class Config:
        orm_mode = True

class PatientBase(BaseModel):
    name: str

class PatientCreate(PatientBase):
    pass

class Patient(PatientBase):
    id: int
    pictures: List[Picture] = []

    class Config:
        orm_mode = True