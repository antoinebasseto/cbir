import React, { useState }  from 'react';
import './App.css';
import { queryBackend } from './backend/BackendQueryEngine';
import Sidebar from "./components/sidebar/sidebar"
import DragDropUploader from './components/dragDropUploader/dragDropUploader';
import XrayDisplay from './components/xrayDisplay/xrayDisplay'

function App() {

  const [file, setFile] = useState(null);
  const [indexActiv, setIndexActiv] = useState(0)
  const [filterActiv, setFilterActiv] = useState(false)


  {/*To be updated with similar images from backend*/}
  const similarImages = [[require("./test1.png"), 1, 0, "Cardiomegaly", 0.94],
                        [require("./test2.png"), 1, 1, "Cardiomegaly|Emphysema", 0.85],
                        [require("./test3.png"), 2, 0, "No Finding", 0.83]]

  function handleUpload(){
      setIndexActiv(0)
  }

  function handleShow(){
      setIndexActiv(1)
  }

  function handleFilter(){
    setFilterActiv(!filterActiv)
  }
  function handleImageUploaded(file) {
      setFile(file);
      setIndexActiv(1); {/*Back to show image.  Not necessary*/}
      {/*TODO: Send image to backend and get similar images*/}
  };

  return (
    <div className="App">
      <div className="container">
        <Sidebar indexActiv={indexActiv} handleUpload={handleUpload} handleShow={handleShow} handleFilter={handleFilter} filterActiv={filterActiv}/>
        <div className="others">
          {indexActiv===0 && <DragDropUploader onImageUploadedChange={handleImageUploaded}/>}
          {indexActiv===1 && file && 
            <XrayDisplay uploadedImageSource={URL.createObjectURL(file)} imgList={similarImages}/> 
        }
        </div>
      </div>
    </div>
  )
}

export default App;
