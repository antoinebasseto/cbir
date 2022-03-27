# Chest XRay CBIR using VAE's latent space

[[_TOC_]]

## Team Members
1. Jieming Li
2. Michael Sommer
3. Loïc Houmard
4. Antoine Basseto

## Project Description 
Our goal is to develop a tool allowing radiologists to leverage advancements in deep learning as new tools for them to work with. More precisely, using a variational auto-encoder's latent space to compute similarities between images, we hope to make possible the retrieval of medically similar XRays to the one the radiologist is studying, in order to take into consideration previous similar cases and their outcomes. This is commonly called content-based image retrieval (CBIR).

### Users
Our target users are therefore domain experts, i.e. radiologists.

### Datasets
The dataset used is the [NIH chest Xrays](https://nihcc.app.box.com/v/ChestXray-NIHCC). Because the dataset is quite big, it is not placed in this repository and you should download it on [their website](https://nihcc.app.box.com/v/ChestXray-NIHCC) and then place it in a folder named "/data".

### Tasks
Our dashboard aims to allow users to:
- Upload a chest XRay they want to study.
- Get back a list of medically relevant similar images, the diagnoses made for them and the computed similarity, as well as to explore and filter that list.
- To view a patient's history and consider their evolution when looking at similar cases, as much as the data available allows us to do.

- - -
## Folder Structure

``` bash
├── README.md  
├── backend-project
│   ├── app.py
│   ├── crud.py
│   ├── database.py
│   ├── models.py
│   ├── data    # data to be downloaded and placed here
│   ├── DL_model
│   │   └── model.py    # class to define deep learning model
│   └── pydantic_models
│       └── schemas.py
├── react-frontend
│   ├── README.md
│   ├── package-lock.json
│   ├── package.json
│   ├── src
│   │   ├── App.css
│   │   ├── App.test.tsx
│   │   ├── App.tsx
│   │   ├── backend
│   │   │   └── BackendQueryEngine.tsx
│   │   ├── components
│   │   │   ├── XRayCBIRComponent.tsx
│   │   │   ├── XRayListComponent.tsx
│   │   │   └── XRayListElementComponent.tsx
│   │   ├── index.css
│   │   ├── index.tsx
│   │   ├── react-app-env.d.ts
│   │   ├── reportWebVitals.ts
│   │   ├── setupTests.ts
│   │   └── types
│   └── tsconfig.json
├── VAE
│   ├── main.py
│   ├── train.py
│   ├── VAE.ipynb
│   ├── models
│   │   └── autoencoder.py
│   └── dataset
│       ├── dataset.py
│       └── utils.py
└── requirements.txt
```

## Requirements & How to Run
To build the environment and run the code:
1. Create a conda environment using the `requirements.txt` file.
2. Run the back-end using `uvicorn app:app --reload`
3. Run the front-end using `npm start`

**NOTE:** To observe our progress on the variational auto-encoder, launch the jupyter notebook present in the `VAE` folder, using the same conda environment.

## Milestones

- [ ] Week 6
  - [ ] Refactor git repo: [#1](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/1) 
  - [ ] Research possible architectures to implement for beta-VAE: [#2](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/2)

## Versioning

- Week 6: [Week 6 Tag]()

