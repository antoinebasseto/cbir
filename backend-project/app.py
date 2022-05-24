from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from numpy import record
import uvicorn
import pandas as pd
import os
import csv
import codecs
from io import StringIO
from typing import Callable
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import base64
import numpy as np
from DL_model.config import config
from DL_model.hyperparameters import params
from DL_model.model import get_model

app = FastAPI(
    title="Test Python Backend",
    description="""This is a template for a Python backend.
                   It provides acess via REST API.""",
    version="0.1.0",
)

model = None
async def get_dl():
    global model
    if not model:
        model = get_model(params[config['model']], config['model'])
    return model


IMAGES_PATH = "./data/images"
METADATA_PATH = "./data/HAM10000_metadata_with_dummy_latent.csv"

ABBREVIATION_TO_DISEASE = {
    "akiec" : "Actinic keratoses and intraepithelial carcinoma",
    "bcc" : "Basal cell carcinoma",
    "bkl" : "Benign keratosis-like lesions",
    "df" : "Dermatofibroma",
    "mel" : "Melanoma",
    "nv" : "Melanocytic nevi",
    "vasc" : "Vascular lesions"
}

similarityThreshold= 0
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
    
    coordinates = metadata[["latent_coordinate" + str(i) for i in range(12)]]
    pca_coordinates = PCA(n_components=2).fit_transform(coordinates)
    projection_data = metadata[["dx", "dx_type", "age", "sex", "localization"]].copy()
    projection_data["x"] = [pca[0] for pca in pca_coordinates]
    projection_data["y"] = [pca[1] for pca in pca_coordinates]
    projection_data = projection_data.fillna("unknown")
    return projection_data.to_dict(orient="records")

@app.get("/get_uploaded_projection_data")
def get_uploaded_projection_data():
    return [{"x": 0, "y": 0}]

# Method used to compute the rollout images for latent space exploration and then send back the path of the generated images
@app.get("/get_latent_space_images_url")
def get_latent_space_images_url():
    # TODO: return the path to the real rollout images to get rollout images
    #honestly this should only be parametrized
    #or cached I guess
    # 10x10 images
    dummy_return = [["ISIC_0024306.jpg", "ISIC_0024307.jpg", "ISIC_0024308.jpg", "ISIC_0024309.jpg", "ISIC_0024310.jpg", "ISIC_0024311.jpg", "ISIC_0024312.jpg", "ISIC_0024313.jpg", "ISIC_0024314.jpg", "ISIC_0024315.jpg"],
                    ["ISIC_0024306.jpg", "ISIC_0024307.jpg", "ISIC_0024308.jpg", "ISIC_0024309.jpg", "ISIC_0024310.jpg", "ISIC_0024311.jpg", "ISIC_0024312.jpg", "ISIC_0024313.jpg", "ISIC_0024314.jpg", "ISIC_0024315.jpg"],
                    ["ISIC_0024306.jpg", "ISIC_0024307.jpg", "ISIC_0024308.jpg", "ISIC_0024309.jpg", "ISIC_0024310.jpg", "ISIC_0024311.jpg", "ISIC_0024312.jpg", "ISIC_0024313.jpg", "ISIC_0024314.jpg", "ISIC_0024315.jpg"],
                    ["ISIC_0024306.jpg", "ISIC_0024307.jpg", "ISIC_0024308.jpg", "ISIC_0024309.jpg", "ISIC_0024310.jpg", "ISIC_0024311.jpg", "ISIC_0024312.jpg", "ISIC_0024313.jpg", "ISIC_0024314.jpg", "ISIC_0024315.jpg"],
                    ["ISIC_0024306.jpg", "ISIC_0024307.jpg", "ISIC_0024308.jpg", "ISIC_0024309.jpg", "ISIC_0024310.jpg", "ISIC_0024311.jpg", "ISIC_0024312.jpg", "ISIC_0024313.jpg", "ISIC_0024314.jpg", "ISIC_0024315.jpg"],
                    ["ISIC_0024306.jpg", "ISIC_0024307.jpg", "ISIC_0024308.jpg", "ISIC_0024309.jpg", "ISIC_0024310.jpg", "ISIC_0024311.jpg", "ISIC_0024312.jpg", "ISIC_0024313.jpg", "ISIC_0024314.jpg", "ISIC_0024315.jpg"],
                    ["ISIC_0024306.jpg", "ISIC_0024307.jpg", "ISIC_0024308.jpg", "ISIC_0024309.jpg", "ISIC_0024310.jpg", "ISIC_0024311.jpg", "ISIC_0024312.jpg", "ISIC_0024313.jpg", "ISIC_0024314.jpg", "ISIC_0024315.jpg"],
                    ["ISIC_0024306.jpg", "ISIC_0024307.jpg", "ISIC_0024308.jpg", "ISIC_0024309.jpg", "ISIC_0024310.jpg", "ISIC_0024311.jpg", "ISIC_0024312.jpg", "ISIC_0024313.jpg", "ISIC_0024314.jpg", "ISIC_0024315.jpg"],
                    ["ISIC_0024306.jpg", "ISIC_0024307.jpg", "ISIC_0024308.jpg", "ISIC_0024309.jpg", "ISIC_0024310.jpg", "ISIC_0024311.jpg", "ISIC_0024312.jpg", "ISIC_0024313.jpg", "ISIC_0024314.jpg", "ISIC_0024315.jpg"],
                    ["ISIC_0024306.jpg", "ISIC_0024307.jpg", "ISIC_0024308.jpg", "ISIC_0024309.jpg", "ISIC_0024310.jpg", "ISIC_0024311.jpg", "ISIC_0024312.jpg", "ISIC_0024313.jpg", "ISIC_0024314.jpg", "ISIC_0024315.jpg"],]
    return dummy_return

@app.get("/image")
def get_image(name: str):
    path = IMAGES_PATH+'/'+name
    return FileResponse(path)

@app.post("/update_filters")
def update_filters(filters: dict):
    print(filters)
    global similarityThreshold, maxNumberImages, ageInterval, diseasesFilter
    similarityThreshold = filters['similarityThreshold']
    maxNumberImages = filters['maxNumberImages']
    ageInterval = filters['ageInterval']
    diseasesFilter = filters['diseasesFilter']
    print(diseasesFilter)
    print(ageInterval[0])
    print(maxNumberImages)
    return True

#todo filters and thresholds
@app.post("/get_similar_images")
def get_similar_images(file: UploadFile = File(...)):
    pictures = pd.read_csv(METADATA_PATH)
    
    #TODO actually use image
    #get dimensions of image in VAE space
    #assuming get_embedding returns tuple/list of dimensions
#    pic_embedding = get_embedding(file, dlmodel)
    pic_embedding = np.random.rand(12)

    #array with dists of uploaded image to saved images
    dists = np.zeros(pictures.shape[0])

    #calculate distance scores for each    
    for i, pic in pictures.iterrows():
        cur_embedding = (pic.values[8:21])
        dists[i] =np.linalg.norm(pic_embedding - cur_embedding)

    pictures["dist"] = dists
    sorted_pictures = (pictures.sort_values(by=['dist']))
    #age
    #filtered_pictures = sorted_pictures[((sorted_pictures['age'] >= ageInterval[0]) & (sorted_pictures['age'] <= ageInterval[1]))]
    filtered_pictures = sorted_pictures[(sorted_pictures['age'] >= ageInterval[0]) & (sorted_pictures['age'] <= ageInterval[0])]
    #threshold
    filtered_pictures = filtered_pictures[filtered_pictures['dist'] > similarityThreshold]
    #diseases
    #filtered_pictures
    closest_pictures = filtered_pictures.iloc[:maxNumberImages]
    result = closest_pictures[["image_id", "lesion_id","dx_type","dx", "dist", "latent_coordinate1","latent_coordinate2","latent_coordinate3","latent_coordinate4","latent_coordinate5","latent_coordinate6","latent_coordinate7","latent_coordinate8","latent_coordinate9","latent_coordinate10","latent_coordinate11"]]
    return_result = result.values.tolist()

    #dummy_return = [["ISIC_0024334.jpg", 1, 0, "Melanocytic nevi", 0.94],
     #                   ["ISIC_0024335.jpg", 1, 1, "Melanocytic nevi", 0.85],
      #                  ["ISIC_0024336.jpg", 2, 0, "Benign keratosis-like lesions", 0.83]]
    return JSONResponse(content=return_result)

@app.post("/files/")
async def create_file(file: bytes = File(...)):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}
    

