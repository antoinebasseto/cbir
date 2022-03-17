from typing import Optional
from sqlalchemy.orm import Session

import models
from pydantic_models import schemas


def get_patient(db: Session, user_id: int):
    return db.query(models.Patient).filter(models.Patient.id == user_id).first()


def get_patients(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Patient).offset(skip).limit(limit).all()


def create_patient(db: Session, patient: schemas.PatientCreate):
    name = patient.name
    db_user = models.Patient(name = name)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_picture(db: Session, picture_id: int):
    return db.query(models.Picture).filter(models.Picture.id == picture_id).first()


def get_picture_by_file_name(db: Session, file_name: str):
    return db.query(models.Picture).filter(models.Picture.title == file_name).first()


def get_pictures(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Picture).offset(skip).limit(limit).all()

#has to be 
def create_picture(db: Session, item, patient_id: Optional[int] = 0):
    db_item = models.Picture(**item)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item
# def create_picture(db: Session, item: schemas.PictureCreate, patient_id: Optional[int] = 0):
#     db_item = models.Picture(**item.dict())
#     db.add(db_item)
#     db.commit()
#     db.refresh(db_item)
#     return db_item