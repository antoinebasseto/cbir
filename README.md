# Chest XRay CBIR using VAE's latent space

[[_TOC_]]

## Team Members
1. Jieming Li
2. Michael Sommer
3. Loïc Houmard
4. Antoine Basseto

## Project Description 
Our goal is to develop a tool allowing doctors to leverage advancements in deep learning as new tools for them to work with. More precisely, using a variational auto-encoder's latent space to compute similarities between images, we hope to make possible the retrieval of medically similar XRays to the one the doctor is studying, in order to take into consideration previous similar cases and their outcomes. This is commonly called content-based image retrieval (CBIR).

### Users
Our target users are therefore domain experts, i.e. doctors.

### Datasets
The dataset used is the  HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Because the dataset is quite big, it is not placed in this repository and you should download it on [their website](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FDBW86T). You only need to download the 3 following files:
- HAM10000_images_part_1.zip
- HAM10000_images_part_2.zip
- HAM10000_metadata.tab
Then, you should create a folder called data inside the backend-project and inside it another folder called images. The two zip files should be unziped in the images folder and the HAM10000\_metadata.tab should be first renamed as HAM10000\_metadata.csv (if not already in csv format) and placed in the data folder. You can see the [Folder Structure](Folder Structure) below for help.


### Tasks
Our dashboard aims to allow users to:
- Upload an image of a skin lesion they want to study.
- Get back a list of medically relevant similar images, the diagnoses made for them and the computed similarity, as well as to explore and filter that list.
- See a projection of this data point and the rest of the dataset into 2D using UMAP.
- Analyze the latent space by seeing a rollout of images made by varying one dimension of at the time.

- - -
## Folder Structure

``` bash
├── README.md  
├── backend-project
│   ├──data # data to be downloaded and placed here
│   │	├── images # The two images.zip files should be unzipped here
│   │	└── HAM10000\_metadata.csv
│   ├── app.py
│   └── ...
│
├── react-frontend
│   ├── README.md
│   ├── package-lock.json
│   ├── package.json
│   ├── src
│   │   └── ...
│   │
│   ├── public
│   │   └── ...
│   │
│   │
│   ├── tsconfig.json
│   └── node_modules # needs to be generated the first time the project is ran, see below
│	└── ...
│
├── VAE
│   ├── main.py
│   ├── train.py
│   ├── run_training.py
│   ├── VAE.ipynb
│   ├── Beta-VAE.ipynb
│   ├── models
│   │   └── ...
│   └── dataset
│       └── ...
│
├── LightningVAE
│   ├── config.py
│   ├── datamodule_factory.py
│   ├── LRP.py
│   ├── rollout.py
│   ├── train.py
│   └── src
│       └── ...
│
└── requirements.txt
```

## Requirements & How to Run
To build the environment and run the code, we use conda. Make sure to have it installed on your computer by checking their [documentation](https://docs.conda.io/en/latest/) and then you can follow the next steps:

1. Create an empty conda environment.
```
conda create --name myenv
```
2. Activate The environment and install the packages using pip and the `requirements.txt` file.
```
conda activate myenv
conda install pip
pip install -r requirements.txt
```
3. Move to the backend directory and run the backend.
```
cd backend-project
uvicorn app:app --reload
```
4. With another terminal, move to the front-end directory. Make sure your conda environment is activated. If it's the first time you run the project, you must create the node_modules directory using the second command (otherwise you can skip this command) and then run the front-end.
```
cd react-frontend
npm install
npm start
```
5. It should open a window in your browser with the app.

**NOTES:** 
* You only need to create the environment, install all the packages using pip and generate the node_modules folder the first time you run the project. Otherwise you can just skip these parts.
* To observe our progress on the variational auto-encoder, launch the jupyter notebook present in the `VAE` folder, using the same conda environment. For that, you can move to the VAE folder and launch a jupyter notebook.
```
cd VAE
conda activate myenv
jupyter notebook
```

## Weekly summary

**Week 7**:
This week, we implemented the first (real) version of the frontend which includes a sidebar, a button to upload an image and a panel to show a list of similar images, which we will update and improve during the following weeks. On the backend, we created a notebook to populate our database. Finally, we transformed our first linear version of the VAE into a convolutional one and kept on searching papers talking about the architecture (to know for example how big our latent space should be).

**Week 10**:
In the previous weeks, a lot of work went into model training, trying different losses, architectures and overall strategies. After discussions with the professor and Lukas Klein, a PhD candidate working on similar subjects, we decided that the current dataset was not appropriate for our task, as no AE could be trained to have an interesting enough latent space. Our decision is therefore to pivot onto a new dataset, the [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000). Previous results by Lukas show that this is a promising lead for our project, notably because of [this paper](https://openreview.net/pdf?id=3uQ2Z0MhnoE). We are therefore adapting our dashboard to this, but expect to keep a CBIR tool, with latent space exploration thanks to rollouts and a projection of the latent space as the main features. Depending on our progress, we hope to implement interpretability features seen in the previously mentionned paper.

## Milestones

- [x] Week 6
  - [x] Refactor git repo: [#1](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/1) 
  - [x] Set up the backbone to train models: [#9](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/9)
  - [x] Update 'Requirements & How to run' instructions in the readme: [#6](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/6)
  - [x] Research beta-VAEs: [#2](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/2)
  - [x] Write a basic VAE: [#3](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/3)
  - [x] Spend time learning basics of React: [#10](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/10)
  - [x] Write the first React components: [#11](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/11)

- [x] Week 7
  - [x] Update frontend with sidebar [#13](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/13), an interface to let the users upload their image [#14](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/14) and display (hard-coded) similar images next to the newly uploaded image [#16](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/16)
  - [x] Write a convolutional VAE [#4](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/4) (but not tested yet)
  - [x] Write code to populate database


- [x] Week 8
  - [x] Finish convolutional VAE, beta-VAE [#4](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/4)[#5](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/5)
  - [x] Struggled with adapting the FactorVAE loss function since different sampling seems to be needed [#8](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/8),
  - [x] Set up environment to train with GPU support[#17](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/17)
  - [x] Implement filters to choose similar images in the frontend (still testing) [#18](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/18)
  - [x] Write code to link database, backend and frontend

- [x] Week 10
  - [x] Paper reading and brainstorming (see Week 10 summary) [#24](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/24)
  - [x] Front-end for latent space projection [#20](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/20)
  - [x] Explore other possible architectures to improve model results [#21](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/21)
  - [x] Almost finish implementing similarity feature [#22](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/22)
  - [x] Added pytorch lightning [#25](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/25)
  - [x] Computed rollouts [#25](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/25)
  - [x] Layerwise relevance computation [#25](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/issues/25)
## Versioning

- Week 6: [Week 6 Tag](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/tags/week6)
- Week 7: [Week 7 Tag](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/tags/week7)
- Week 8: [Week 8 Tag](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/tags/week8)
- Week 10: [Week 10 Tag](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/tags/week10)
