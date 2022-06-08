from typing import Callable, List, Any
import io
import os
import torch
from PIL import Image
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
from model.config import config
from model.hyperparameters import params
from model.model import get_model, get_image_preprocessor, rollout

IMAGES_PATH = config['image_path']
METADATA_PATH = config['metadata_path']
CACHE_DIR = config['cache_dir']
ABBREVIATION_TO_DISEASE = {
    "akiec" : "Actinic keratoses and intraepithelial carcinoma",
    "bcc" : "Basal cell carcinoma",
    "bkl" : "Benign keratosis-like lesions",
    "df" : "Dermatofibroma",
    "mel" : "Melanoma",
    "nv" : "Melanocytic nevi",
    "vasc" : "vascular lesions"
}

app = FastAPI(
    title="Skinterpret's Backend",
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

distanceWeights = [1,1,1,1,1,1,1,1,1,1,1,1]
maxNumberImages = 3
ageInterval = [0, 100]
diseasesFilter = ["All"]

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    return pic_embedding.tolist()


@app.get("/get_projection_data")
def get_projection_data():
    metadata = pd.read_csv(METADATA_PATH)

    projection_data = metadata[["image_id", "dx", "dx_type", "age", "sex", "localization", "umap1", "umap2"]].copy()
    projection_data = projection_data.fillna("unknown")

    return projection_data.to_dict(orient="records")


@app.get("/get_uploaded_projection_data")
def get_uploaded_projection_data(latent):
    latent = str(latent).strip('[]').strip(']').split(',')
    pic_embedding = np.array(latent, dtype=np.float32)
    reducer = joblib.load("data/umap.sav")
    embedding = reducer.transform(pic_embedding.reshape(1, -1))
    return [{"umap1": embedding[0][0].item(), "umap2": embedding[0][1].item()}]


# Method used to compute the rollout images for latent space exploration and then send back the path of the generated images
@app.get("/get_latent_space_images_url")
def get_latent_space_images_url(latent, model=Depends(get_dl)):

    # Empty the cache
    for file in os.scandir(CACHE_DIR):
        os.remove(file.path)
    
    latent = str(latent).strip('[]').strip(']').split(',')
    latent_space = np.array(latent, dtype=np.float32)
    latent_space = torch.from_numpy(latent_space).view(1, -1)
    ret = rollout(model, latent_space, CACHE_DIR, -5, 6, 11) #We use 6 (not 5) since the upper bound is not included and we want a symmetric interval
    return ret


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
    global distanceWeights, maxNumberImages, ageInterval, diseasesFilter
    distanceWeights = filters['distanceWeights']
    maxNumberImages = filters['maxNumberImages']
    ageInterval = filters['ageInterval']
    diseasesFilter = filters['diseasesFilter']
    return True


@app.get("/get_similar_images")
def get_similar_images(latent):
    latent = str(latent).strip('[]').strip(']').split(',')
    pic_embedding = np.array(latent, dtype=np.float32)

    pictures = pd.read_csv(METADATA_PATH)
    pictures["dx"] = pictures["dx"].apply(lambda x: ABBREVIATION_TO_DISEASE[x])
    
    latents = pictures.loc[:, [f"latent_coordinate_{i}" for i in range(12)]]
    pictures["dist"] = (latents - pic_embedding).multiply(distanceWeights).apply(np.linalg.norm, ord=1, axis=1)
    sorted_pictures = pictures.sort_values(by=['dist'])

    filtered_pictures = sorted_pictures[(sorted_pictures['age'] >= ageInterval[0]) & (sorted_pictures['age'] <= ageInterval[1])]
    if not "All" in diseasesFilter:
        filtered_pictures = filtered_pictures[filtered_pictures["dx"].isin(diseasesFilter)]
    
    closest_pictures = filtered_pictures.iloc[:maxNumberImages]

    return closest_pictures.to_dict(orient="records")


def update_schema_name(app: FastAPI, function: Callable, name: str) -> None:
    for route in app.routes:
        print(route)
        if route.endpoint is function:
            print(route.body_field)
            route.body_field.type_.__name__ = name
            break


update_schema_name(app, get_latent_space, "get_similar_images")
#update_schema_name(app, get_uploaded_projection_data, "get_uploaded_data")
