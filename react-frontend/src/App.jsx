import { useState, useEffect }  from 'react';
import './App.css';
import { queryBackend , queryBackendWithFile, updateFiltersBackend } from './backend/BackendQueryEngine';
import Sidebar from "./components/sidebar/sidebar"
import DragDropUploader from './components/dragDropUploader/dragDropUploader';
import SimilarImages from './components/similarImages/similarImages'
import ProjectionPlot from './components/projectionPlot/projectionPlot';
import LatentSpaceExplorator from './components/latentSpaceExplorator/latentSpaceExplorator';

function App() {

  const [file, setFile] = useState(null);
  const [indexActiv, setIndexActiv] = useState(0)
  const [filterActiv, setFilterActiv] = useState(false)

  /* Filters */
  const [similarityThreshold, setSimilarityThreshold] = useState(90)
  const [maxNumberImages, setMaxNumberImages] = useState(3)
  const [ageInterval, setAgeInterval] = useState([0, 85])
  const [diseasesFilter, setDiseasesFilter] = useState(['All'])

  {/*To be updated with similar images from backend*/}
  // const similarImages = [[require("./test1.png"), 1, 0, "Cardiomegaly", 0.94],
  //                       [require("./test2.png"), 1, 1, "Cardiomegaly|Emphysema", 0.85],
  //                       [require("./test3.png"), 2, 0, "No Finding", 0.83]]
  const [similarImages, setSimilarImages] = useState([]);
  const [projectionData, setProjectionData] = useState([]);
  const [uploadedProjectionData, setUploadedProjectionData] = useState([]);
  const [latentSpaceExplorationImages, setLatentSpaceExplorationImages] = useState([]);

  useEffect(() => {
      queryBackend('get_projection_data', 'GET').then((data) => {
        setProjectionData(data)
      })
    }, []);
  
  function handleUpload(){
    setIndexActiv(0)
  }

  function handleShow() {
    setIndexActiv(1)
  }

  function handleShowProjection() {
    setIndexActiv(2)
  }

  function handleShowExplore() {
    setIndexActiv(3)
  }

  function handleFilter() {
    setFilterActiv(!filterActiv)
  }

  function handleImageUploaded(file) {
    setFile(file);
    setIndexActiv(1); /*Back to show image.*/
    
    
    queryBackendWithFile('get_similar_images', file).then((data) => {
        setSimilarImages(data)
      }
    )
    
    // Get uploaded image projection data
    queryBackendWithFile('get_uploaded_projection_data', file).then((data) => {
        setUploadedProjectionData(data)
      }
    )

    // Get the rollout images for latent space exploration
    queryBackend('get_latent_space_images_url', 'GET').then((latent_space_images_url) => {
        setLatentSpaceExplorationImages(latent_space_images_url)
      }
    )
  };

  function applyOnClickHandle() {
    updateFiltersBackend('update_filters', 'POST', similarityThreshold, maxNumberImages, ageInterval, diseasesFilter)
    if (file !== null)
    	handleImageUploaded(file)
  }

  return (
    <div className="App">
      <div className="container">
        <Sidebar indexActiv={indexActiv} handleUpload={handleUpload} handleShow={handleShow} handleFilter={handleFilter} filterActiv={filterActiv} 
                handleShowProjection={handleShowProjection} handleShowExplore={handleShowExplore}
                similarityThreshold={similarityThreshold} maxNumberImages={maxNumberImages} ageInterval={ageInterval} diseasesFilter={diseasesFilter}
                setDiseasesFilter = {setDiseasesFilter} setSimilarityThreshold={setSimilarityThreshold} setMaxNumberImages={setMaxNumberImages} 
                setAgeInterval={setAgeInterval} applyOnClickHandle={applyOnClickHandle}/>
        <div className="others">
          {indexActiv===0 && <DragDropUploader onImageUploadedChange={handleImageUploaded}/>}
          {indexActiv===1 && file && 
            <SimilarImages uploadedImageSource={URL.createObjectURL(file)} imgList={similarImages}/> 
          }
          {indexActiv===2 && <ProjectionPlot data={projectionData} uploadedData={uploadedProjectionData}/>}
          {indexActiv===3 && <LatentSpaceExplorator uploadedImage={file} latentSpaceImagesPath={latentSpaceExplorationImages}/>}
        </div>
      </div>
    </div>
  )
}

export default App;
