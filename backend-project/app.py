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

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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

@app.post("/upload-picture", response_model=schemas.Picture)
def upload_picture(file:str, db: Session = Depends(get_db)):
    picture = crud.get_picture_by_file_name(db, file)
    if picture:
        raise HTTPException(status_code=400, detail="file already exists")
    picture = {'title': "test", 
               'file_path': "dataset/test.png"}
    return crud.create_picture(db=db, item=picture)

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



@app.post("/embedding")
def get_embedding(file: bytes = File(...)):
    """
    model callback function
    @param file:
    @return:
    """
    image_embedding = get_embedding(file, dlmodel)
    # Add possible annotations?
    return image_embedding

@app.post("/get-dataset")
def get_data(name: str, db: Session = Depends(get_db)):
    image = crud.get_picture_by_file_name(db, file_name="test")
    imagepath = image.file_path

    with open(imagepath, 'rb') as f:
    	base64image = base64.b64encode(f.read())
    	return base64image

# @app.post("/get-dataset")
# def get_data(name: str):
#     with open(imagepath, 'rb') as f:
#     	base64image = base64.b64encode(f.read())
#     	return base64image

@app.get("/image_ids")
def image_list(db: Session = Depends(get_db)):
    return {"image_ids": crud.picture_ids(db)}

@app.get("/image")
def get_image(name: str):
    path = PICTURE_FOLDER+'/'+name
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

# Upload an image, calculate score and return similar files
#To be finalized once dataset/database is decided

#@app.post("/uploadfile2/")
#async def create_upload_file2(file: UploadFile, picture_schema, threshold):
    #get dimensions of image in VAE space
    #assuming get_embedding returns tuple/list of dimensions
#    pic_embedding = get_embedding(file, dlmodel)
    #query DB for all images (filtered)
#    pictures = crud.filter_pictures(picture_schema)

    #array with dists
#    dists = np.array(len(pictures))

    #calculate distance scores for each    
#    for i, pic in enumerate(pictures):
#        cur_embedding = [pic.Dim1, pic.Dim2, pic.Dim3, pic.Dim4, pic.Dim5]
#        dists[i] =np.linalg.norm(pic_embedding - cur_embedding)
    #threshold distance array, take corresponding pictures
#    thresh_pics = pictures[dists<threshold]
#    smal_dists =dists[dists<threshold]
    #return smallest distance
#    thresh_pics[np.argsort(small_dists)[-10:]]
    #todo based on final DB layout
#    dummy_return = [["test1.png", 1, 0, "Cardiomegaly", 0.94],
#     ["test2.png", 1, 1, "Cardiomegaly|Emphysema", 0.85],
#     ["test3.png", 2, 0, "No Finding", 0.83]]
#   return JSONResponse(content=dummy_return)


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
