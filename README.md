# Skin lesions CBIR using VAE's latent space

[[_TOC_]]

## Team Members
1. Jieming Li
2. Michael Sommer
3. Loïc Houmard
4. Antoine Basseto

## Project Description 
Our goal is to develop a tool allowing doctors to leverage advancements in deep learning as new tools for them to work with. More precisely, using a variational auto-encoder's latent space to compute similarities between images, we hope to make possible the retrieval of medically similar skin lesions to the one the doctor is studying, in order to take into consideration previous similar cases and their outcomes. This is commonly called content-based image retrieval (CBIR).

### Users
Our target users are therefore domain experts, i.e. doctors, dermatologists in particular.

### Datasets
The dataset used is the  HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Because the dataset is quite big, it is not placed in this repository and you should download it on [their website](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FDBW86T).

See [Requirements](Requirements) for instructions on what to download and where to put it.


### Workflow
- Upload an image of a skin lesion they want to study.
- Use the Explore dimensions menu to see what each dimension represents. Each row represents the impact on the image of varying one dimension while keeping the others untouched. You can rename a dimension if it represents a particular feature or you think it is important to remember it.
- Use the weights in the Filter menu to give more or less importance to a dimension while computing the distance between two images. By default all dimensions are weighted equally. You can also filter the results based on some criterion such as a given disease or age range.
- Look at what images are considered similar and their corresponding diagnoses under the menu Similar images.
- Look at the Projection menu to have an overview of the full dataset in 2D. The similar images and your uploaded image will be highlighted so that you can easily find them. You can have more information about one image by putting your mouse over it in the visualization.


- - -
## Folder Structure

``` bash
├── README.md  
├── backend-project
│   ├── data # This folder (and therefore all those it contains) has to be created
│   │	├── images
│   │	│    └── # The two zip files containing the images should be unzipped here
│   │	├── cache # Empty folder where latent space exploration images will be cached, you need to create it
│   │	├── umap.sav # You need to download this
│   │	└── HAM10000_latent_space_umap_processed.csv # You need to download this
|   ├── model
│   │	├── epoch=61-step=9734.ckpt # You need to download this
|   │ └── ...
│   ├── app.py
│   └── ...
│
├── react-frontend
│   ├── README.md
│   ├── package-lock.json
│   ├── package.json
│   ├── src
│   │   └── ...
│   ├── public
│   │   └── ...
│   ├── tsconfig.json
│   ├── node_modules # You need to generate this
│	  └── ...
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

## Requirements
Before running the project for the first time, you will need to do a couple of things:
1. Clone the current repository somewhere on your laptop.
2. Download the HAM10000 dataset from [their website](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FDBW86T). Create a folder called `backend-project/data` and inside it a folder called `backend-project/data/images` and unzip the images (HAM10000\_images\_part\_1.zip  and HAM10000\_images\_part\_2.zip) inside it.
3. Download `HAM10000_latent_space_umap_processed.csv`, `umap.sav` and `epoch=61-step=9734.ckpt` from [polybox](https://polybox.ethz.ch/index.php/s/5qdTV1qiaAo35K3). Put the csv and sav files in `backend-project/data`, and the ckpt file in `backend-project/model`.
4. Create the following empty folder `backend-project/data/cache`.
5. Create an empty conda environment.

    ```
    conda create --name skin-cbir python==3.10
    ```
6. Activate the environment, and install required packages using pip and the `requirements.txt` file.

    ```
    conda activate skin-cbir
    pip install -r requirements.txt
    ```
7. Move to the front-end folder and install its requirements.

    ```
    cd react-frontend
    npm install
    npm install react-svg-radar-chart
    ```

## How to run
To run the project, follow the instructions below:
1. Make sure you've completed the [Requirements](Requirements) listed in the corresponding section.
2. Activate the conda environment and run the backend.

    ```
    conda activate skin-cbir
    cd backend-project
    uvicorn app:app --reload
    ```
3. In another terminal window, run the frontend.

    ```
    cd react-frontend
    npm start
    ```
4. The app should then automatically open in your browser. Please note that on rare occasions uploading an image will fail and nothing will happen, this can be easily seen by not having the loading animation present in the Explore dimensions menu. If such a thing happens, simply reuploading the picture should work. Also note that we happened to have some linking errors between our frontend and backend when using Firefox as a browser, which didn't occured when trying with other browsers.


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
- Final submission: [Final Submission Tag](https://gitlab.inf.ethz.ch/COURSE-XAI-IML22/Medical1-xai-iml22/-/tags/final-submission)
