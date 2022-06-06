from typing import Callable, List, Any
# db dependencies
# from sqlalchemy.orm import Session
# from db import models
# from pydantic_models import schemas
# from db.database import SessionLocal, engine
import io

import torch
from PIL import Image
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

from fastapi.middleware.cors import CORSMiddleware
import joblib

import pandas as pd
import numpy as np
# import umap.umap_ as umap

from model.config import config
from model.hyperparameters import params
from model.model import get_model, get_image_preprocessor, rollout

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


IMAGES_PATH = config['image_path']
METADATA_PATH = config['metadata_path']

similarityThreshold = 0
distanceWeights = [1,1,1,1,1,1,1,1,1,1,1,1]
maxNumberImages = 3
ageInterval = [0, 100]
diseasesFilter = ["All"]

ABBREVIATION_TO_DISEASE = {
    "akiec" : "Actinic keratoses and intraepithelial carcinoma",
    "bcc" : "Basal cell carcinoma",
    "bkl" : "Benign keratosis-like lesions",
    "df" : "Dermatofibroma",
    "mel" : "Melanoma",
    "nv" : "Melanocytic nevi",
    "vasc" : "vascular lesions"
}


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


@app.get("/get_uploaded_projection_data")
def get_uploaded_projection_data(latent, model=Depends(get_dl),
                                       preprocess=Depends(get_image_processor)):

    latent = str(latent).strip('[]').strip(']').split(',')
    #latent = str(latent).split(",")
    print('latent', latent)

    pic_embedding = np.array(latent, dtype=np.float32)
    reducer = joblib.load("umap.sav")
    embedding = reducer.transform(pic_embedding.reshape(1, -1))
    return [{"umap1": embedding[0][0].item(), "umap2": embedding[0][1].item()}]


# Method used to compute the rollout images for latent space exploration and then send back the path of the generated images
@app.get("/get_latent_space_images_url")
def get_latent_space_images_url(latent ,model=Depends(get_dl), preprocess=Depends(get_image_processor)):
    # TODO: return the path to the real rollout images to get rollout images
    # honestly this should only be parametrized
    # or cached I guess
    # 10x10 images
    # try:
    #     contents = await file.read()
    #     img = Image.open(io.BytesIO(contents))
    # except:
    #     return {"message": "Error uploading file"}
    # finally:
    #     await file.close()
    # img = preprocess(img)
    # img = img.unsqueeze(0)
    # pic_embedding, _ = model.encoder(img)
    # latent_space = pic_embedding.detach()
    print(latent)
    latent = str(latent).strip('[]').strip(']').split(',')
    # latent = str(latent).split(",")
    print('latent', latent)
    latent_space = np.zeros(12, dtype=np.float32)
    latent_space = torch.from_numpy(latent_space).view(1, -1)
    print(latent_space)
    ret = rollout(model, latent_space, config['cache_dir'], -5, 5, 10)
    return ret

CACHE_DIR = config['cache_dir']

@app.get("/cache")
def get_rollout_image(name: str):
    path = f"{CACHE_DIR}/{name}"
    return FileResponse(path)


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
    print(diseasesFilter)
    return True


# todo filters and thresholds

@app.post("/get_latent_space")
async def get_latent_space(file: UploadFile = File(...), model=Depends(get_dl),
                           preprocess=Depends(get_image_processor)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
    except:
        return {"message": "Error uploading file"}
    finally:
        await file.close()
    img = preprocess(img)
    img = img.unsqueeze(0)
    pic_embedding, _ = model.encoder(img)
    pic_embedding = pic_embedding.squeeze().detach().numpy()
    print(pic_embedding.tolist())
    return pic_embedding.tolist()


@app.get("/get_similar_images")
def get_similar_images(latent, model=Depends(get_dl),
                             preprocess=Depends(get_image_processor)):
    # async def get_similar_images(file: UploadFile):
    pictures = pd.read_csv(METADATA_PATH)
    #
    #     #TODO actually use image
    #     #get dimensions of image in VAE space
    # #
    latent = str(latent).strip('[]').strip(']').split(',')
    # latent = str(latent).split(",")
    print('latent', latent)
    pic_embedding = np.array(latent, dtype=np.float32)
    pictures["dist"] = (pictures.loc[:, [f"latent_coordinate_{i}" for i in range(12)]] - pic_embedding).apply(
        np.linalg.norm, axis=1)

    # print(pic_embedding)
    # pic_embedding = np.random.rand(12)
    # Calculate distance scores for each
    #pictures = pd.read_csv("HAM10000_metadata_with_dummy_latent.csv")
    pictures["dx"]= pictures["dx"].apply(lambda x: ABBREVIATION_TO_DISEASE[x])
    latents = pictures.loc[:, [f"latent_coordinate{i}" for i in range(12)]]
    weighted_latents =latents.multiply(distanceWeights)
    pictures["dist"] = (weighted_latents - pic_embedding).apply(np.linalg.norm, ord=1, axis=1)

    sorted_pictures = (pictures.sort_values(by=['dist']))
    filtered_pictures = sorted_pictures[(sorted_pictures['age'] >= ageInterval[0]) & (sorted_pictures['age'] <= ageInterval[0])]
    if not "All" in diseasesFilter:
        filtered_pictures = filtered_pictures[filtered_pictures["dx"].isin(diseasesFilter)]
    #filtered_pictures = filtered_pictures[filtered_pictures['dist'] > similarityThreshold]
    closest_pictures = filtered_pictures.iloc[:maxNumberImages]

    return JSONResponse(content=closest_pictures.values.tolist())


def update_schema_name(app: FastAPI, function: Callable, name: str) -> None:
    for route in app.routes:
        print(route)
        if route.endpoint is function:
            print(route.body_field)
            route.body_field.type_.__name__ = name
            break


update_schema_name(app, get_latent_space, "get_latent_space")
# update_schema_name(app, get_uploaded_projection_data, "get_uploaded_data")
