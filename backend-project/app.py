
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Callable
import base64
#db dependencies
from sqlalchemy.orm import Session
import crud
from db import models
from pydantic_models import schemas
from db.database import SessionLocal, engine
from DL_model.model import get_model, get_embedding
import io

from PIL import Image
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

from fastapi.middleware.cors import CORSMiddleware
import joblib

import pandas as pd
import numpy as np
#import umap.umap_ as umap

from model.config import config
from model.hyperparameters import params
from model.model import get_model, get_image_preprocessor

app = FastAPI(
    title="Test Python Backend",
    description="""This is a template for a Python backend.
                   It provides acess via REST API.""",
    version="0.1.0",
)



model = None
image_preprocessor = None

def get_dl():
    global model
    if not model:
        print(model)
        model = get_model(params[config['model']], config['model'], config['results_dir'], config['model_path'])
    return model

def get_image_processor():
    global image_preprocessor
    if not image_preprocessor:
        image_preprocessor = get_image_preprocessor(params[config['model']], config['model'])
    return image_preprocessor

IMAGES_PATH = "/home/jimmy/Medical1-xai-iml22/LightningVAE/data/HAM10000/images"
METADATA_PATH = "/home/jimmy/Medical1-xai-iml22/LightningVAE/HAM10000_latent_space_umap_processed.csv"

similarityThreshold = 0
maxNumberImages = 3
ageInterval = [0,100]
diseasesFilter = ["All"]

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/get_projection_data")
def get_projection_data():
    metadata = pd.read_csv(METADATA_PATH)

    projection_data = metadata[["image_id", "dx", "dx_type", "age", "sex", "localization", "umap1", "umap2"]].copy()
    projection_data = projection_data.fillna("unknown")
    
    return projection_data.to_dict(orient="records")

@app.post("/get_uploaded_projection_data")
async def get_uploaded_projection_data(file: UploadFile = File(...), model = Depends(get_dl), preprocess = Depends(get_image_processor)):
    try:
        #contents = await file.read()
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
    except:
        return {"message": "Error uploading file"}
    finally:
        await file.close()
        #await file.close()
    img = preprocess(img)
    img = img.unsqueeze(0)
    pic_embedding, _ = model.encoder(img)
    #print(pic_embedding)
    #pic_embedding = np.random.rand(12)
    pic_embedding = pic_embedding.squeeze().detach().numpy()
    reducer = joblib.load("umap.sav")
    embedding = reducer.transform(pic_embedding.reshape(1, -1))
    return [{"umap1": embedding[0][0].item(), "umap2": embedding[0][1].item()}]

# Method used to compute the rollout images for latent space exploration and then send back the path of the generated images
@app.get("/get_latent_space_images_url")
def get_latent_space_images_url():
    # TODO: return the path to the real rollout images to get rollout images
    #honestly this should only be parametrized
    #or cached I guess
    # 10x10 images
    dummy_return = [["ISIC_0024306", "ISIC_0024307", "ISIC_0024308", "ISIC_0024309", "ISIC_0024310", "ISIC_0024311", "ISIC_0024312", "ISIC_0024313", "ISIC_0024314", "ISIC_0024315"],
                    ["ISIC_0024306", "ISIC_0024307", "ISIC_0024308", "ISIC_0024309", "ISIC_0024310", "ISIC_0024311", "ISIC_0024312", "ISIC_0024313", "ISIC_0024314", "ISIC_0024315"],
                    ["ISIC_0024306", "ISIC_0024307", "ISIC_0024308", "ISIC_0024309", "ISIC_0024310", "ISIC_0024311", "ISIC_0024312", "ISIC_0024313", "ISIC_0024314", "ISIC_0024315"],
                    ["ISIC_0024306", "ISIC_0024307", "ISIC_0024308", "ISIC_0024309", "ISIC_0024310", "ISIC_0024311", "ISIC_0024312", "ISIC_0024313", "ISIC_0024314", "ISIC_0024315"],
                    ["ISIC_0024306", "ISIC_0024307", "ISIC_0024308", "ISIC_0024309", "ISIC_0024310", "ISIC_0024311", "ISIC_0024312", "ISIC_0024313", "ISIC_0024314", "ISIC_0024315"],
                    ["ISIC_0024306", "ISIC_0024307", "ISIC_0024308", "ISIC_0024309", "ISIC_0024310", "ISIC_0024311", "ISIC_0024312", "ISIC_0024313", "ISIC_0024314", "ISIC_0024315"],
                    ["ISIC_0024306", "ISIC_0024307", "ISIC_0024308", "ISIC_0024309", "ISIC_0024310", "ISIC_0024311", "ISIC_0024312", "ISIC_0024313", "ISIC_0024314", "ISIC_0024315"],
                    ["ISIC_0024306", "ISIC_0024307", "ISIC_0024308", "ISIC_0024309", "ISIC_0024310", "ISIC_0024311", "ISIC_0024312", "ISIC_0024313", "ISIC_0024314", "ISIC_0024315"],
                    ["ISIC_0024306", "ISIC_0024307", "ISIC_0024308", "ISIC_0024309", "ISIC_0024310", "ISIC_0024311", "ISIC_0024312", "ISIC_0024313", "ISIC_0024314", "ISIC_0024315"],
                    ["ISIC_0024306", "ISIC_0024307", "ISIC_0024308", "ISIC_0024309", "ISIC_0024310", "ISIC_0024311", "ISIC_0024312", "ISIC_0024313", "ISIC_0024314", "ISIC_0024315"],]
    return dummy_return

@app.get("/image")
def get_image(name: str):
    path = f"{IMAGES_PATH}/{name}.jpg"
    return FileResponse(path)

@app.post("/update_filters")
def update_filters(filters: dict):
    global similarityThreshold, maxNumberImages, ageInterval, diseasesFilter
    similarityThreshold = filters['similarityThreshold']
    maxNumberImages = filters['maxNumberImages']
    ageInterval = filters['ageInterval']
    diseasesFilter = filters['diseasesFilter']
    return True

#todo filters and thresholds

@app.post("/get_similar_images")
async def get_similar_images( file: UploadFile = File(...), model = Depends(get_dl), preprocess = Depends(get_image_processor)):
#async def get_similar_images(file: UploadFile):
    print(file)
    pictures = pd.read_csv(METADATA_PATH)
#
#     #TODO actually use image
#     #get dimensions of image in VAE space
#     #assuming get_embedding returns tuple/list of dimensions
#
#    pic_embedding = get_embedding(file, dlmodel)
    try:
        #contents = await file.read()
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
    except:
        return {"message": "Error uploading file"}
    finally:
        await file.close()
        #await file.close()
    img = preprocess(img)
    img = img.unsqueeze(0)
    pic_embedding, _ = model.encoder(img)
    #print(pic_embedding)
    #pic_embedding = np.random.rand(12)
    pic_embedding = pic_embedding.squeeze().detach().numpy()
    # Calculate distance scores for each 
    pictures["dist"] = (pictures.loc[:, [f"latent_coordinate_{i}" for i in range(12)]] - pic_embedding).apply(np.linalg.norm, axis=1)
    sorted_pictures = (pictures.sort_values(by=['dist']))
    filtered_pictures = sorted_pictures[(sorted_pictures['age'] >= ageInterval[0]) & (sorted_pictures['age'] <= ageInterval[0])]
    filtered_pictures = filtered_pictures[filtered_pictures['dist'] > similarityThreshold]
    closest_pictures = filtered_pictures.iloc[:maxNumberImages]
    
    return JSONResponse(content=closest_pictures.values.tolist())

# @app.post("/files/")
# async def create_file(file: bytes = File(...)):
#     return {"file_size": len(file)}


# @app.post("/uploadfile/")
# async def create_upload_file(file: UploadFile):
#     return {"filename": file.filename}

def update_schema_name(app: FastAPI, function: Callable, name: str) -> None:

    for route in app.routes:
        print(route)
        if route.endpoint is function:
            print(route.body_field)
            route.body_field.type_.__name__ = name
            break

update_schema_name(app, get_similar_images, "get_similar_images")
update_schema_name(app, get_uploaded_projection_data, "get_uploaded_data")