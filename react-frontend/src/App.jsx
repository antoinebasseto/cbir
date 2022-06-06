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

  // Dependent on uploaded image
  const [similarImages, setSimilarImages] = useState([]);
  const [projectionData, setProjectionData] = useState([]);
  const [uploadedProjectionData, setUploadedProjectionData] = useState([]);
  const [latentSpaceExplorationImages, setLatentSpaceExplorationImages] = useState([]);

  const [latent_space, setLatentSpace] = useState([0,0,0,0,0,0,0,0,0,0,0,0]);
  useEffect(() => {
      queryBackend('get_projection_data', 'GET').then((data) => {
        setProjectionData(data)
      })
    }, []);
  
  function handleUpload(){
    setIndexActiv(0)
  }

  function handleShow() {
    if (!file) return
    setIndexActiv(1)
  }

  function handleShowProjection() {
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
    //Get latent space
    queryBackendWithFile('get_latent_space', file).then((data) => {
      setLatentSpace(data);
      console.log(data)
    });

    console.log(latent_space)
    // Get uploaded image projection data
    queryBackend(`get_uploaded_projection_data?latent=[${latent_space}]`, 'GET').then((data) => {

        setUploadedProjectionData(data)
      }
    );
    queryBackend(`get_similar_images?latent=[${latent_space}]`, 'GET').then((data) => {
        setSimilarImages(data)
      }
    )
    // Get the rollout images for latent space exploration

    queryBackend(`get_latent_space_images_url?latent=[${latent_space}]`, 'GET').then((latent_space_images_url) => {
        setLatentSpaceExplorationImages(latent_space_images_url)
    })
  };

  function applyOnClickHandle() {
    updateFiltersBackend('update_filters', 'POST', similarityThreshold, maxNumberImages, ageInterval, diseasesFilter)
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
          setDiseasesFilter={setDiseasesFilter} 
          setSimilarityThreshold={setSimilarityThreshold} 
          setMaxNumberImages={setMaxNumberImages} 
          setAgeInterval={setAgeInterval} 
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
            <ProjectionPlot data={projectionData} uploadedData={uploadedProjectionData} similarImages={similarImages.map(a => a[1]).flat()}/>
          }
          {
            indexActiv===3 && 
            <LatentSpaceExplorator uploadedImage={file} latentSpaceImagesPath={latentSpaceExplorationImages}/>
          }
        </div>
      </div>
    </div>
  )
}

export default App;
