import { useState, useEffect } from "react";
import "./App.css";
import {
  queryBackend,
  queryBackendWithFile,
} from "./backend/BackendQueryEngine";
import SimilarImages from "./components/similarImages/similarImages";
import ProjectionPlot from "./components/projectionPlot";
import LatentSpaceExplorator from "./components/latentSpaceExplorator";

function App() {
  // Navigation
  const [demo, setDemo] = useState("choice");
  const [numDim, setNumDim] = useState(0);

  // Projection
  const [projectionData, setProjectionData] = useState([]);
  const [labels, setLabels] = useState([]);
  const [dataPointFocused, setDataPointFocused] = useState(null);

  // Rollout
  const [isGeneratingRollout, setIsGeneratingRollout] = useState(false);
  const [latentSpaceExplorationImages, setLatentSpaceExplorationImages] =
    useState([]);
  const [latentSpaceExplorationNames, setLatentSpaceExplorationNames] =
    useState([]);
  const [isGeneratingRolloutProjection, setIsGeneratingRolloutProjection] =
    useState(false);
  const [explorationProjectionData, setExplorationProjectionData] = useState([
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
  ]);
  const [
    dimensionExplorationProjectionFocused,
    setDimensionExplorationProjectionFocused,
  ] = useState(0);
  const [rolloutClustering, setRolloutClustering] = useState([]);
  const [focusedLatent, setFocusedLatent] = useState([]);

  // Dependent on uploaded image
  const [file, setFile] = useState(null);
  const [similarImages, setSimilarImages] = useState([]);
  const [uploadedLatent, setUploadedLatent] = useState([]);
  const [uploadedProjectionData, setUploadedProjectionData] = useState([]);

  // Filters
  const [distanceWeights, setDistanceWeights] = useState([]);

  useEffect(() => {
    if (demo != "choice") {
      var nd;
      if (demo === "skin") {
        nd = 12;
      } else if (demo === "mnist") {
        nd = 10;
      }
      setNumDim(nd);
      setDistanceWeights(Array(nd).fill(0.5));
      setLatentSpaceExplorationNames(
        Array.from([...Array(nd).keys()], (i) => `Dim ${i}`)
      );
      setExplorationProjectionData(Array(nd).fill([]));

      queryBackend(`get_projection_data?demo=${demo}`, "GET").then((data) => {
        setLabels(data["labels"]);
        setProjectionData(data["data"]);
      });
    }
  }, [demo]);

  function handleClickOnDataPoint(datapoint) {
    setDataPointFocused(datapoint);
    setRolloutClustering([]);
    setIsGeneratingRollout(true);
    setIsGeneratingRolloutProjection(true);
    setExplorationProjectionData(Array(numDim).fill([]));

    var latent = [];
    for (const [key, value] of Object.entries(datapoint)) {
      if (key.startsWith("latent_coordinate")) {
        latent.push(value);
      }
    }
    setFocusedLatent(latent);

    // Cluster rollouts
    queryBackend(
      `get_rollout_clustering?demo=${demo}&weights=${distanceWeights}&latent=[${latent}]`,
      "GET"
    ).then((data) => {
      setRolloutClustering(data);
    });

    // Get the rollout images for latent space exploration
    queryBackend(
      `get_latent_space_images_url?demo=${demo}&latent=[${latent}]`,
      "GET"
    ).then((latent_space_images_url) => {
      setLatentSpaceExplorationImages(latent_space_images_url);
      setIsGeneratingRollout(false);
    });

    // Generate projection of rollout images
    queryBackend(
      `get_projection_rollout?demo=${demo}&latent=[${latent}]`,
      "GET"
    ).then((data) => {
      setExplorationProjectionData(data);
      setIsGeneratingRolloutProjection(false);
    });
  }

  function handleImageUploaded(file) {
    setUploadedLatent([]);
    setSimilarImages([]);
    setRolloutClustering([]);
    setIsGeneratingRollout(true);
    setIsGeneratingRolloutProjection(true);
    setExplorationProjectionData(Array(numDim).fill([]));

    setFile(file);

    // Get latent space amd projection of uploaded file
    queryBackendWithFile(
      `get_uploaded_latent_and_projection/${demo}`,
      file
    ).then((data) => {
      // Add to projection
      setUploadedProjectionData([{ umap1: data.umap1, umap2: data.umap2 }]);

      // Act as if it was clicked on (e.g. generate rollout)
      handleClickOnDataPoint(data);

      // Get similar images
      var latent = [];
      for (const [key, value] of Object.entries(data)) {
        if (key.startsWith("latent_coordinate")) {
          latent.push(value);
        }
      }
      setUploadedLatent(latent);
      queryBackend(
        `get_similar_images/${demo}?weights=${distanceWeights}&latent=[${latent}]`,
        "GET"
      ).then((data) => {
        setSimilarImages(data);
      });
    });
  }

  function handleRenameLatent(event, dim) {
    let temp = [...latentSpaceExplorationNames];
    temp[dim] = event.target.value;
    setLatentSpaceExplorationNames(temp);
  }

  function handleClickShowBackProjection(dimensionNumber) {
    if (dimensionExplorationProjectionFocused != dimensionNumber) {
      setDimensionExplorationProjectionFocused(dimensionNumber);
    }
  }

  function handleFilterWeightsChange(newValue, dim) {
    setRolloutClustering([]);
    let temp = [...distanceWeights];
    temp[dim] = newValue;
    setDistanceWeights(temp);

    // Update cluster rollouts
    queryBackend(
      `get_rollout_clustering?demo=${demo}&weights=${temp}&latent=[${focusedLatent}]`,
      "GET"
    ).then((data) => {
      setRolloutClustering(data);
    });

    // Update similar images
    if (file !== null) {
      queryBackend(
        `get_similar_images/${demo}?weights=${temp}&latent=[${uploadedLatent}]`,
        "GET"
      ).then((data) => {
        setSimilarImages(data);
      });
    }
  }

  return (
    <div>
      {demo === "choice" && (
        <div className="grid grid-rows-2 w-screen h-screen place-items-center">
          <div
            className="bg-transparent hover:bg-blue-500 text-blue-700 font-semibold hover:text-white py-2 px-4 border border-blue-500 hover:border-transparent rounded cursor-pointer"
            onClick={function () {
              setDemo("mnist");
            }}
          >
            MNIST
          </div>
          <div
            className="bg-transparent hover:bg-blue-500 text-blue-700 font-semibold hover:text-white py-2 px-4 border border-blue-500 hover:border-transparent rounded cursor-pointer"
            onClick={function () {
              setDemo("skin");
            }}
          >
            Skin Lesions
          </div>
        </div>
      )}
      {demo != "choice" && (
        <div className="grid grid-rows-2">
          <div className="p-4 grid grid-cols-2 items-center w-screen h-screen">
            <ProjectionPlot
              data={projectionData}
              labels={labels}
              handleClickOnDataPoint={handleClickOnDataPoint}
              uploadedData={uploadedProjectionData}
              similarImages={similarImages}
              explorationData={explorationProjectionData}
              demo={demo}
              dimensionExplorationProjectionFocused={
                dimensionExplorationProjectionFocused
              }
              rolloutClustering={rolloutClustering}
            />

            {dataPointFocused != null && (
              <LatentSpaceExplorator
                className="col-span-3"
                demo={demo}
                dataPointFocused={dataPointFocused}
                latentSpaceImagesPath={latentSpaceExplorationImages}
                dimensionNames={latentSpaceExplorationNames}
                handleRenameLatent={handleRenameLatent}
                isGeneratingRollout={isGeneratingRollout}
                isGeneratingRolloutProjection={isGeneratingRolloutProjection}
                handleClickShowBackProjection={handleClickShowBackProjection}
                rolloutClustering={rolloutClustering}
                distanceWeights={distanceWeights}
                handleFilterWeightsChange={handleFilterWeightsChange}
                dimensionExplorationProjectionFocused={
                  dimensionExplorationProjectionFocused
                }
              />
            )}
          </div>

          <div className="grid grid-rows-2 grid-cols-2 place-items-center w-screen h-screen">
            {/* <DragDropUploader handleImageUploaded={handleImageUploaded}/> */}
            <input
              type="file"
              accept="image/*"
              onChange={(e) => {
                console.log(e);
                handleImageUploaded(e.target.files[0]);
              }}
            />

            {file && <img className="w-52" src={URL.createObjectURL(file)} />}

            {similarImages.length > 0 && (
              <div className="col-span-2">
                <SimilarImages
                  uploadedImageSource={URL.createObjectURL(file)}
                  imgList={similarImages}
                  dimensionNames={latentSpaceExplorationNames}
                  latentSpace={uploadedLatent}
                  demo={demo}
                />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
