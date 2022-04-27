from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import os
import csv
import codecs
from io import StringIO
from typing import Callable
from sklearn.cluster import KMeans
import base64
#db dependencies
from sqlalchemy.orm import Session
import crud, models
from pydantic_models import schemas
from database import SessionLocal, engine
from DL_model.model import get_model, get_embedding


models.Base.metadata.create_all(bind=engine)

dlmodel = get_model()

app = FastAPI(
    title="Test Python Backend",
    description="""This is a template for a Python backend.
                   It provides acess via REST API.""",
    version="0.1.0",
)

PICTURE_FOLDER = "/home/jj/Spring2022/Medical1-xai-iml22/backend-project/image_folder"

#Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

imagepath = "./test.png"

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# @app.post("/upload-data")
# def upload_data(file: UploadFile = File(...)):
#     ##data = pd.read_csv(file.file)
#     ##if cluster_algo
#     print(file)
#     data = pd.read_csv(file.file).to_dict()
#     print(data)
#     return data


@app.post("/upload-picture", response_model=schemas.Picture)
def upload_picture(file:str, db: Session = Depends(get_db)):
    picture = crud.get_picture_by_file_name(db, file)
    if picture:
        raise HTTPException(status_code=400, detail="file already exists")
    picture   = {'title': "test", 
                'file_path': "data/test.png"}
    return crud.create_picture(db =db, item=picture)
    
#@app.post("/get-data", response_class=FileResponse)
#def get_data(name: str):
#    return FileResponse(imagepath)

@app.post("/embedding")
def get_embedding(file: bytes = File(...)):
    """
    model callback function
    @param file:
    @return:
    """
    image_embedding = get_embedding(file, dlmodel)
    #add possible annotations?
    
    return image_embedding

@app.post("/get-data")
def get_data(name: str, db: Session = Depends(get_db)):
    image = crud.get_picture_by_file_name(db, file_name="test")
    imagepath = image.file_path

    with open(imagepath, 'rb') as f:
    	base64image = base64.b64encode(f.read())
    	return base64image

# @app.post("/get-data")
# def get_data(name: str):
#     with open(imagepath, 'rb') as f:
#     	base64image = base64.b64encode(f.read())
#     	return base64image

@app.get("/image_ids")
def image_list(db: Session = Depends(get_db)):
    return {"image_ids": crud.picture_ids(db)}

@app.get("/image")
def get_image(name: str, db: Session = Depends(get_db)):
    # image = crud.get_picture(db, id)
    # file_name = image.picture_id
    path = PICTURE_FOLDER+'/'+name
    #path = "/home/jj/Spring2022/Medical1-xai-iml22/backend-project/test.png"
    return FileResponse(path = path)

@app.post("/query")
def get_array(id: str, db: Session = Depends(get_db)):

    # dummy_return = [[ 0, 1, 0, "Cardiomegaly", 0.94],
    #                     [ 1, 1, 1, "Cardiomegaly|Emphysema", 0.85],
    #                     [2, 2, 0, "No Finding", 0.83]]
    dummy_return = [["test1.png", 1, 0, "Cardiomegaly", 0.94],
     ["test2.png", 1, 1, "Cardiomegaly|Emphysema", 0.85],
     ["test3.png", 2, 0, "No Finding", 0.83]]
    return JSONResponse(content=dummy_return)

@app.post("/files/")
async def create_file(file: bytes = File(...)):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}



@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
        <html>
            <head>
                <title>Week 2</title>
            </head>
            <body>
                <h1>Test Python Backend</h1>
                Visit the <a href="/docs">API doc</a> (<a href="/redoc">alternative</a>) for usage information.
            </body>
        </html>
        """
    return HTMLResponse(content=html_content, status_code=200)


def update_schema_name(app: FastAPI, function: Callable, name: str) -> None:
    """
    Updates the Pydantic schema name for a FastAPI function that takes
    in a fastapi.UploadFile = File(...) or bytes = File(...).

    This is a known issue that was reported on FastAPI#1442 in which
    the schema for file upload routes were auto-generated with no
    customization options. This renames the auto-generated schema to
    something more useful and clear.

    Args:
        app: The FastAPI application to modify.
        function: The function object to modify.
        name: The new name of the schema.
    """
    for route in app.routes:
        if route.endpoint is function:
            route.body_field.type_.__name__ = name
            break


update_schema_name(app, create_file, "CreateFileSchema")
update_schema_name(app, create_upload_file, "CreateUploadSchema")
