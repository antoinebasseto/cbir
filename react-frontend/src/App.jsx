import { useState, useEffect, useRef, useLayoutEffect }  from 'react';
import './App.css';
import { queryBackend, queryBackendWithFile, updateFiltersBackend } from './backend/BackendQueryEngine';
import DragDropUploader from './components/dragDropUploader/dragDropUploader';
import SimilarImages from './components/similarImages/similarImages'
import ProjectionPlot from './components/projectionPlot/projectionPlot';
import LatentSpaceExplorator from './components/latentSpaceExplorator';



function App() {

    // Navigation
    const [file, setFile] = useState(null);
    const [filterActiv, setFilterActiv] = useState(false)

    // Filters
    const [similarityThreshold, setSimilarityThreshold] = useState(90)
    const [maxNumberImages, setMaxNumberImages] = useState(3)
    const [ageInterval, setAgeInterval] = useState([0, 85])
    const [diseasesFilter, setDiseasesFilter] = useState(['All'])
    const [distanceWeights, setDistanceWeights] = useState([1,1,1,1,1,1,1,1,1,1,1,1])

    // Dependent on uploaded image
    const [similarImages, setSimilarImages] = useState([]);
    const [uploadedLatent, setUploadedLatent] = useState([]);
    const [uploadedProjectionData, setUploadedProjectionData] = useState([]);

    // Projection
    const [projectionData, setProjectionData] = useState([]);
    const [dataPointFocused, setDataPointFocused] = useState(null);

    // Rollout
    const [isGeneratingRollout, setIsGeneratingRollout] = useState(false);
    const [latentSpaceExplorationImages, setLatentSpaceExplorationImages] = useState([]);
    const [latentSpaceExplorationNames, setLatentSpaceExplorationNames] = useState(
        ['Dim 1', 'Dim 2', 'Dim 3', 'Dim 4', 'Dim 5', 'Dim 6', 'Dim 7', 'Dim 8', 'Dim 9', 'Dim 10', 'Dim 11', 'Dim 12']);
    const [isGeneratingRolloutProjection, setIsGeneratingRolloutProjection] = useState(false)
    const [explorationProjectionData, setExplorationProjectionData] = useState([
        {"dim": 0, "data": []}, {"dim": 1, "data": []}, {"dim": 2, "data": []}, {"dim": 3, "data": []},
        {"dim": 4, "data": []}, {"dim": 5, "data": []}, {"dim": 6, "data": []}, {"dim": 7, "data": []},
        {"dim": 8, "data": []}, {"dim": 9, "data": []}, {"dim": 10, "data": []}, {"dim": 11, "data": []}]);
    const [dimensionExplorationProjectionFocused, setDimensionExplorationProjectionFocused] = useState(0)
    const [rolloutClustering, setRolloutClustering] = useState([])
    

    useEffect(() => {
        // This code will run once
        queryBackend('get_projection_data', 'GET').then((data) => {
            setProjectionData(data)
        })
    }, [])


    function handleClickOnDataPoint(datapoint) {
        setDataPointFocused(datapoint)
        const latent = [datapoint.latent_coordinate_0, datapoint.latent_coordinate_1, datapoint.latent_coordinate_2, datapoint.latent_coordinate_3, datapoint.latent_coordinate_4, datapoint.latent_coordinate_5, datapoint.latent_coordinate_6, datapoint.latent_coordinate_7, datapoint.latent_coordinate_8, datapoint.latent_coordinate_9, datapoint.latent_coordinate_10, datapoint.latent_coordinate_11]

        // Get the rollout images for latent space exploration
        setIsGeneratingRollout(true)
        queryBackend(`get_latent_space_images_url?latent=[${latent}]`, 'GET').then((latent_space_images_url) => {
            setLatentSpaceExplorationImages(latent_space_images_url)
            setIsGeneratingRollout(false)
        })

        // Cluster rollouts
        queryBackend(`get_rollout_clustering?latent=[${latent}]`, 'GET').then((data) => {
            setRolloutClustering(data)
        })

        // Generate projection of rollout images
        setIsGeneratingRolloutProjection(true)
        queryBackend(`get_projection_rollout?latent=[${latent}]`, 'GET').then((data) => {
            setExplorationProjectionData(data)
            setIsGeneratingRolloutProjection(false)
        })
    }

    function handleImageUploaded(file) {
        setFile(file);

        // Get latent space amd projection of uploaded file
        queryBackendWithFile('get_uploaded_latent_and_projection', file).then((data) => {

            // Add to projection
            setUploadedProjectionData([{"umap1": data.umap1, "umap2": data.umap2}])

            // Act as if it was clicked on (e.g. generate rollout)
            handleClickOnDataPoint(data)

            // Get similar images 
            setUploadedLatent([data.latent_coordinate_0, data.latent_coordinate_1, data.latent_coordinate_2, data.latent_coordinate_3, data.latent_coordinate_4, data.latent_coordinate_5, data.latent_coordinate_6, data.latent_coordinate_7, data.latent_coordinate_8, data.latent_coordinate_9, data.latent_coordinate_10, data.latent_coordinate_11])
            queryBackend(`get_similar_images?latent=[${[data.latent_coordinate_0, data.latent_coordinate_1, data.latent_coordinate_2, data.latent_coordinate_3, data.latent_coordinate_4, data.latent_coordinate_5, data.latent_coordinate_6, data.latent_coordinate_7, data.latent_coordinate_8, data.latent_coordinate_9, data.latent_coordinate_10, data.latent_coordinate_11]}]`, 'GET').then((data) => {
                setSimilarImages(data)
            })
        });
    };

    function handleRenameLatent(event, dim) {
        let temp = [...latentSpaceExplorationNames]
        temp[dim] = event.target.value
        setLatentSpaceExplorationNames(temp)
    }

    function handleMouseEnterRow(dimensionNumber) {
        // if (!isGeneratingRolloutProjection) {
        //     setDimensionExplorationProjectionFocused(dimensionNumber)
        //     console.log(`dim number: ${dimensionNumber}`)
        //     console.log(`should display dim: ${explorationProjectionData[0].data}`)
        // }
    }

    function handleMouseLeaveRow(dimensionNumber) {
        // setDimensionExplorationProjectionFocused(12)
    }

    function handleFilterWeightsChange(newValue, dim) {
        let temp = [...distanceWeights]
        temp[dim] = newValue
        setDistanceWeights(temp)
    }

    // function applyOnClickHandle() {
    //     updateFiltersBackend('update_filters', 'POST', distanceWeights, maxNumberImages, ageInterval, diseasesFilter)
    //     if (file !== null){
    //     // Get latent space
    //     queryBackendWithFile('get_latent_space', file).then((data) => {
    //         setLatentSpace(data);
    //         console.log(data)

    //         // Get similar images 
    //         queryBackend(`get_similar_images?latent=[${data}]`, 'GET').then((data) => {
    //             setSimilarImages(data)
    //         })

    //         // Get uploaded image projection data
    //         queryBackend(`get_uploaded_projection_data?latent=[${data}]`, 'GET').then((data) => {
    //             setUploadedProjectionData(data)
    //         })
    //     });    
    //     }
    // }

    return (
        <div className="grid grid-rows-2">
            <div className="p-8 grid grid-cols-2 items-center w-screen h-screen">
                <ProjectionPlot
                    data={projectionData}
                    handleClickOnDataPoint={handleClickOnDataPoint}
                    uploadedData={uploadedProjectionData} 
                    similarImages={similarImages}
                    explorationData={explorationProjectionData}
                    dimensionExplorationProjectionFocused={dimensionExplorationProjectionFocused}
                />

                {
                    dataPointFocused != null &&
                    <LatentSpaceExplorator
                        dataPointFocused={dataPointFocused} 
                        latentSpaceImagesPath={latentSpaceExplorationImages} 
                        dimensionNames={latentSpaceExplorationNames} 
                        handleRenameLatent={handleRenameLatent} 
                        isGeneratingRollout={isGeneratingRollout}
                        handleMouseEnterRow={handleMouseEnterRow}
                        handleMouseLeaveRow={handleMouseLeaveRow}
                        rolloutClustering={rolloutClustering}
                    />
                }
            </div>

            <div className="grid grid-rows-2 w-screen h-screen">
                <DragDropUploader onImageUploadedChange={handleImageUploaded}/>

                {
                    similarImages.length > 0 &&
                    <SimilarImages
                        uploadedImageSource={URL.createObjectURL(file)} 
                        imgList={similarImages}
                        dimensionNames={latentSpaceExplorationNames} 
                        latentSpace={uploadedLatent}
                    />
                }
            </div>
        </div>
    )
}

export default App;
