import React, { useState, useEffect }  from 'react';
import './App.css';
import { queryBackend } from './backend/BackendQueryEngine';
import Sidebar from "./components/sidebar/sidebar"
import DragDropUploader from './components/dragDropUploader/dragDropUploader';
import XrayDisplay from './components/xrayDisplay/xrayDisplay'

function App() {

  const [file, setFile] = useState(null);
  const [indexActiv, setIndexActiv] = useState(0)
  const [filterActiv, setFilterActiv] = useState(false)

  /* Filters */
  const [similarityThreshold, setSimilarityThreshold] = useState(90)
  const [maxNumberImages, setMaxNumberImages] = useState(3)
  const [followUpInterval, setFollowUpInterval] = useState([1, 10])
  const [diseasesFilter, setDiseasesFilter] = useState(['All'])

  {/*To be updated with similar images from backend*/}
  // const similarImages = [[require("./test1.png"), 1, 0, "Cardiomegaly", 0.94],
  //                       [require("./test2.png"), 1, 1, "Cardiomegaly|Emphysema", 0.85],
  //                       [require("./test3.png"), 2, 0, "No Finding", 0.83]]
  const [similarImages, update_images] = useState([]);

  useEffect(() => {queryBackend('query?id=0').then((exampleData)=>
    {update_images(exampleData)})});

  console.log(similarImages)

  function handleUpload(){
      setIndexActiv(0)
  }

  function handleShow(){
      setIndexActiv(1)
  }

  function handleShowProjection(){
    setIndexActiv(2)
  }

  function handleShowExplore(){
    setIndexActiv(3)
  }

  function handleFilter(){
    setFilterActiv(!filterActiv)
  }
  function handleImageUploaded(file) {
      setFile(file);
      setIndexActiv(1); {/*Back to show image.  Not necessary*/}
      {/*TODO: Send image to backend and get similar images*/}
  };

  function applyOnClickHandle(){
    console.log("Apply filters")
    {/* TODO: call backend to retrieve images with given filters */}
  }

  return (
    <div className="App">
      <div className="container">
        <Sidebar indexActiv={indexActiv} handleUpload={handleUpload} handleShow={handleShow} handleFilter={handleFilter} filterActiv={filterActiv} 
                handleShowProjection={handleShowProjection} handleShowExplore={handleShowExplore}
                similarityThreshold={similarityThreshold} maxNumberImages={maxNumberImages} followUpInterval={followUpInterval} diseasesFilter={diseasesFilter}
                setDiseasesFilter = {setDiseasesFilter} setSimilarityThreshold={setSimilarityThreshold} setMaxNumberImages={setMaxNumberImages} 
                setFollowUpInterval={setFollowUpInterval} applyOnClickHandle={applyOnClickHandle}/>
        <div className="others">
          {indexActiv===0 && <DragDropUploader onImageUploadedChange={handleImageUploaded}/>}
          {indexActiv===1 && file && 
            <XrayDisplay uploadedImageSource={URL.createObjectURL(file)} imgList={similarImages}/> 
          }
          {indexActiv===2 && <div>Here will be a Umap of the latent space</div>}
          {indexActiv===3 && <div>Here will be a tool to explore the different dimensions of the uploaded image</div>}

        </div>
      </div>
    </div>
  )
}

export default App;
