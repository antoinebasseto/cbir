import warnings
import io
import os
import torch
from PIL import Image
from fastapi import FastAPI, Depends, UploadFile, File
from fastapi.responses import FileResponse
import torchvision.transforms as T
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from model.config import config
from model.hyperparameters import params
from model.model import get_model, get_image_preprocessor, rollout, rollout_i

from disentangling_vae.disvae.utils.modelIO import load_model

IMAGES_PATH = config['image_path']
METADATA_PATH = config['metadata_path']
CACHE_DIR = config['cache_dir']
PROJECTOR_DIR = config['projector_path']

NUM_DIM = config['num_dim']

ABBREVIATION_TO_DISEASE = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "vascular lesions"
}

app = FastAPI(
    # title="Skinterpret's Backend",
    # description="""This is a template for a Python backend.
    #                It provides acess via REST API.""",
    # version="0.1.0",
)

# model = None
# image_preprocessor = None


def get_dl(demo: str):
    if demo == "skin":
        return get_model(params[config['model'][demo]], config['model'][demo], config['results_dir'], config['model_path'][demo])
    elif demo == "mnist":
        return load_model(config['model_path'][demo])


def get_preprocessor(demo: str):
    if demo == "skin":
        return get_image_preprocessor(params[config['model'][demo]], config['model'][demo])
    elif demo == "mnist":
        return T.Compose([T.Resize((32, 32)), T.ToTensor()])


# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


############################
# PROJECTION
############################


@app.get("/get_projection_data")
def get_projection_data(demo: str):
    metadata = pd.read_csv(METADATA_PATH[demo])
    metadata = metadata.fillna("unknown")
    metadata['label'] = metadata['label'].astype(str)
    return {"data": metadata.to_dict(orient="records"), 'labels': list(np.sort(metadata['label'].unique()).astype(str))}


@app.get("/image")
def get_image(demo: str, name: str):
    path = f"{IMAGES_PATH[demo]}/{name}.jpg"
    return FileResponse(path)


# ROLLOUT


@app.get("/get_latent_space_images_url")
def get_latent_space_images_url(latent: str, model=Depends(get_dl)):

    # Empty the cache
    for file in os.scandir(CACHE_DIR):
        os.remove(file.path)

    latent = str(latent).strip('[]').strip(']').split(',')
    latent = np.array(latent, dtype=np.float32)
    latent = torch.from_numpy(latent).view(1, -1)
    return rollout(model, latent, CACHE_DIR, -5, 6, 11)


@app.get("/get_rollout_clustering")
def get_rollout_clustering(demo: str, weights: str, latent: str):
    latent = str(latent).strip('[]').strip(']').split(',')
    latent = np.array(latent, dtype=np.float32)
    latent = torch.from_numpy(latent).view(1, -1)

    weights = np.array(str(weights).split(','), dtype=np.float32)
    def metric(a, b):
        return np.sum((a - b)**2 * weights)

    metadata = pd.read_csv(METADATA_PATH[demo])
    neigh = KNeighborsClassifier(n_neighbors=3, metric=metric, weights="distance")
    neigh.fit(metadata[[f'latent_coordinate_{i}' for i in range(NUM_DIM[demo])]].values, metadata['label'])

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        rollout_points = np.empty((NUM_DIM[demo], 11, NUM_DIM[demo]))
        for i in range(NUM_DIM[demo]):
            rollout_points[i] = rollout_i(latent, i, 11, 6, -5)
        ret = neigh.predict(rollout_points.reshape(NUM_DIM[demo]*11, NUM_DIM[demo])).reshape(NUM_DIM[demo], 11).astype(str).tolist()

    return ret


@app.get("/get_projection_rollout")
def get_rollout_projection(demo: str, latent: str):
    latent = str(latent).strip('[]').strip(']').split(',')
    latent = np.array(latent, dtype=np.float32)
    latent = torch.from_numpy(latent).view(1, -1)

    reducer = joblib.load(PROJECTOR_DIR[demo])

    rollout_points = np.empty((NUM_DIM[demo], 11, NUM_DIM[demo]))
    for i in range(NUM_DIM[demo]):
        rollout_points[i] = np.array(rollout_i(latent, i, 11, 6, -5), dtype=np.float32)

    embeddings = reducer.transform(rollout_points.reshape(NUM_DIM[demo]*11, NUM_DIM[demo])).reshape((NUM_DIM[demo], 11, 2)).tolist()

    return [[{"umap1": emb[0], "umap2": emb[1]} for emb in embedding_dim] for embedding_dim in embeddings]


@app.get("/cache")
def get_rollout_image(name: str):
    path = f"{CACHE_DIR}/{name}"
    return FileResponse(path)


# SIMILAR IMAGES


@app.post("/get_uploaded_latent_and_projection/{demo}")
async def get_uploaded_latent_and_projection(
    demo: str,
    file: UploadFile = File(...),
    model=Depends(get_dl),
    preprocess=Depends(get_preprocessor)
):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
    except:
        return {"message": "Error uploading file"}
    finally:
        await file.close()

    # Get latent
    img = preprocess(img)
    img = img.unsqueeze(0)
    pic_embedding, _ = model.encoder(img)
    pic_embedding = pic_embedding.squeeze().detach().numpy()

    # Project
    reducer = joblib.load(PROJECTOR_DIR[demo])
    embedding = reducer.transform(pic_embedding.reshape(1, -1))

    # Return
    ret = {}
    latent = pic_embedding.tolist()
    for i, l in enumerate(latent):
        ret[f"latent_coordinate_{i}"] = l
    ret["umap1"] = embedding[0][0].item()
    ret["umap2"] = embedding[0][1].item()
    return ret


@app.get("/get_similar_images/{demo}")
def get_similar_images(demo: str, weights: str, latent: str):
    latent = str(latent).strip('[]').strip(']').split(',')
    weights = np.array(str(weights).split(','), dtype=np.float32)

    pic_embedding = np.array(latent, dtype=np.float32)

    pictures = pd.read_csv(METADATA_PATH[demo])

    if demo == 'skin':
        pictures["label"] = pictures["label"].apply(lambda x: ABBREVIATION_TO_DISEASE[x])

    latents = pictures.loc[:, [f"latent_coordinate_{i}" for i in range(NUM_DIM[demo])]]

    pictures["dist"] = np.sum((latents - pic_embedding)**2 * weights, axis=1)
    sorted_pictures = pictures.sort_values(by=['dist'])

    closest_pictures = sorted_pictures.iloc[:3].copy()

    closest_latents = closest_pictures.loc[:, [f"latent_coordinate_{i}" for i in range(NUM_DIM[demo])]]
    latent_distances = np.abs(closest_latents - pic_embedding)
    maxval = latent_distances.to_numpy().max()
    latent_distances /= maxval

    for i in range(NUM_DIM[demo]):
        closest_pictures[f"latent_distance_{i}"] = latent_distances.T.iloc[i]

    return closest_pictures.to_dict(orient="records")


# def update_schema_name(app: FastAPI, function: Callable, name: str) -> None:
#     for route in app.routes:
#         print(route)
#         if route.endpoint is function:
#             print(route.body_field)
#             route.body_field.type_.__name__ = name
#             break


# update_schema_name(app, get_uploaded_latent_and_projection, "get_similar_images")
# update_schema_name(app, get_uploaded_projection_data, "get_uploaded_data")
