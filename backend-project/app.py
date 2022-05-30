from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
import umap.umap_ as umap

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
        print(model)
        model = get_model(params[config[model]], config['model'], config['results_dir'], config['model_path'])
        print(model)
    return model


IMAGES_PATH = "./data/images"
METADATA_PATH = "./data/HAM10000_metadata_with_dummy_latent.csv"

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
    
    coordinates = metadata[["latent_coordinate" + str(i) for i in range(12)]]
    global reducer
    reducer = umap.UMAP(random_state = 0)
    reducer = reducer.fit(coordinates)

    embeddings = reducer.transform(coordinates)

    projection_data = metadata[["dx", "dx_type", "age", "sex", "localization"]].copy()
    projection_data = projection_data.fillna("unknown")

    projection_data["x"] = [embedding[0] for embedding in embeddings]
    projection_data["y"] = [embedding[1] for embedding in embeddings]
    
    return projection_data.to_dict(orient="records")

@app.post("/get_uploaded_projection_data")
def get_uploaded_projection_data(file: UploadFile):
    embedding = reducer.transform([['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']])
    print(embedding)
    return [{"x": embedding[0][1], "y": embedding[0][1]}]

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
def get_similar_images(file: UploadFile):
    pictures = pd.read_csv(METADATA_PATH)
    
    #TODO actually use image
    #get dimensions of image in VAE space
    #assuming get_embedding returns tuple/list of dimensions
    # pic_embedding = get_embedding(file, dlmodel)
    uploaded_image_embedding = np.random.rand(12)

    # Calculate distance scores for each 
    pictures["dist"] = (pictures.iloc[:, 7:] - uploaded_image_embedding).apply(np.linalg.norm, axis=1)
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
    

