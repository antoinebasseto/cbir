import { useState, useEffect }  from 'react';
import './App.css';
import { queryBackend , queryBackendWithFile, updateFiltersBackend } from './backend/BackendQueryEngine';
import Sidebar from "./components/sidebar/sidebar"
import DragDropUploader from './components/dragDropUploader/dragDropUploader';
import SimilarImages from './components/similarImages/similarImages'
import ProjectionPlot from './components/projectionPlot/projectionPlot';
import LatentSpaceExplorator from './components/latentSpaceExplorator/latentSpaceExplorator';



function App() {

  // Navigation
  const [file, setFile] = useState(null);
  const [indexActiv, setIndexActiv] = useState(0)
  const [filterActiv, setFilterActiv] = useState(false)

  // Filters
  const [similarityThreshold, setSimilarityThreshold] = useState(90)
  const [maxNumberImages, setMaxNumberImages] = useState(3)
  const [ageInterval, setAgeInterval] = useState([0, 85])
  const [diseasesFilter, setDiseasesFilter] = useState(['All'])
  const [distanceWeights, setDistanceWeights] = useState([1,1,1,1,1,1,1,1,1,1,1,1])

  // Dependent on uploaded image
  const [similarImages, setSimilarImages] = useState([]);
  const [projectionData, setProjectionData] = useState([]);
  const [uploadedProjectionData, setUploadedProjectionData] = useState([]);
  const [latentSpaceExplorationImages, setLatentSpaceExplorationImages] = useState([]);
  const [latentSpaceExplorationNames, setLatentSpaceExplorationNames] = useState([]);
  const [latent_space, setLatentSpace] = useState([0,0,0,0,0,0,0,0,0,0,0,0]);  

  function handleRenameLatent(event, dim){
    let temp = latentSpaceExplorationNames.map((x) => x) // We do that to copy the array
    temp[dim] = event.target.value
    setLatentSpaceExplorationNames(temp)
  }

  function handleUpload(){
    setIndexActiv(0)
  }

  function handleShow() {
    if (!file) return
    setIndexActiv(1)
  }

  function handleShowProjection() {
    if (projectionData.length == 0) {
      queryBackend('get_projection_data', 'GET').then((data) => {
        setProjectionData(data)
      })
    }
    setIndexActiv(2)
  }

  function handleShowExplore() {
    if (!file) return
    setIndexActiv(3)
  }

  function handleFilter() {
    if (!file) return
    setFilterActiv(!filterActiv)
  }

  function handleFilterWeightsChange(newValue, dim){
    let temp = distanceWeights.map((x) => x) // We do that to copy the array
    temp[dim] = newValue
    setDistanceWeights(temp)
  }

  function handleImageUploaded(file0) {
    setFile(file0);
    // queryBackendWithFile('get_latent_space_images_url', file).then((data) => {
    //     setLatentSpaceExplorationImages(data)
    //   }
    // // )
    // queryBackendWithFile('get_similar_images', file).then((data) => {
    //     setSimilarImages(data)
    //   }
    // )
    // Get latent space
    queryBackendWithFile('get_latent_space', file).then((data) => {
      setLatentSpace(data);
      console.log(data)
    });

    // Get uploaded image projection data
    queryBackend(`get_uploaded_projection_data?latent=[${latent_space}]`, 'GET').then((data) => {
        setUploadedProjectionData(data)
      }
    );

    console.log('gonna get sim images')
    queryBackend(`get_similar_images?latent=[${latent_space}]`, 'GET').then((data) => {
        setSimilarImages(data)
        similarImages.map((i) => {console.log(i); return 0})
      }
    )

    
    // Get the rollout images for latent space exploration
    queryBackend(`get_latent_space_images_url?latent=[${latent_space}]`, 'GET').then((latent_space_images_url) => {
        setLatentSpaceExplorationImages(latent_space_images_url)
        
        if(latentSpaceExplorationNames.length === 0){
          let initialNames = latent_space_images_url.map((_, index) => {
            return "Dim " + (index + 1)
          })
          setLatentSpaceExplorationNames(initialNames)
        }
    })
  };

  function applyOnClickHandle() {
    updateFiltersBackend('update_filters', 'POST', distanceWeights, maxNumberImages, ageInterval, diseasesFilter)
    if (file !== null)
    	handleImageUploaded(file)
  }

  return (
    <div className="App">
      <div className="container">
        <Sidebar 
          file={file}
          indexActiv={indexActiv} 
          handleUpload={handleUpload} 
          handleShow={handleShow} 
          handleFilter={handleFilter} 
          filterActiv={filterActiv} 
          handleShowProjection={handleShowProjection} 
          handleShowExplore={handleShowExplore}
          similarityThreshold={similarityThreshold} 
          maxNumberImages={maxNumberImages} 
          ageInterval={ageInterval} 
          diseasesFilter={diseasesFilter}
          distanceWeights={distanceWeights}
          latentSpaceExplorationNames = {latentSpaceExplorationNames}
          setDiseasesFilter={setDiseasesFilter} 
          setSimilarityThreshold={setSimilarityThreshold} 
          setMaxNumberImages={setMaxNumberImages} 
          setAgeInterval={setAgeInterval} 
          handleFilterWeightsChange={handleFilterWeightsChange}
          applyOnClickHandle={applyOnClickHandle}
        />
        
        <div className="others">
          {
            indexActiv===0 && 
            <DragDropUploader onImageUploadedChange={handleImageUploaded}/>
          }
          {
            indexActiv===1 && 
            file && 
            <SimilarImages uploadedImageSource={URL.createObjectURL(file)} imgList={similarImages}/> 
          }
          {
            indexActiv===2 && 
            <ProjectionPlot data={projectionData} uploadedData={uploadedProjectionData} similarImages={similarImages}/>
          }
          {
            indexActiv===3 && 
            <LatentSpaceExplorator uploadedImage={file} latentSpaceImagesPath={latentSpaceExplorationImages} dimensionNames={latentSpaceExplorationNames} handleRenameLatent={handleRenameLatent}/>
          }
        </div>
      </div>
    </div>
  )
}

export default App;
