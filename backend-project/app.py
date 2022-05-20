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

app = FastAPI(
    title="Test Python Backend",
    description="""This is a template for a Python backend.
                   It provides acess via REST API.""",
    version="0.1.0",
)

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

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/projection")
def get_projection_data():
    metadata = pd.read_csv(METADATA_PATH)
    
    coordinates = metadata[["latent_coordinate" + str(i) for i in range(12)]]
    pca_coordinates = PCA(n_components=2).fit_transform(coordinates)
    projection_data = metadata[["dx", "dx_type", "age", "sex", "localization"]].copy()
    projection_data["x"] = [pca[0] for pca in pca_coordinates]
    projection_data["y"] = [pca[1] for pca in pca_coordinates]
    projection_data = projection_data.fillna("NA")
    return projection_data.to_dict(orient="records")

# Method used to compute the rollout images for latent space exploration and then send back the path of the generated images
@app.get("/get_latent_space_images_url")
def get_latent_space_images_url():
    # TODO: return the path to the real rollout images to get rollout images
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

@app.post("/get_similar_images")
def get_similar_images():
    #TODO get real similar images
    dummy_return = [["ISIC_0024334.jpg", 1, 0, "Melanocytic nevi", 0.94],
                    ["ISIC_0024335.jpg", 1, 1, "Melanocytic nevi", 0.85],
                    ["ISIC_0024336.jpg", 2, 0, "Benign keratosis-like lesions", 0.83]]
    return JSONResponse(content=dummy_return)

@app.post("/files/")
async def create_file(file: bytes = File(...)):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}