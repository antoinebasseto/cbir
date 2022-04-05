import React, { useState }  from 'react';
import './App.css';
import { queryBackend } from './backend/BackendQueryEngine';
import XRayCBIR from './components/xrayCBIR/XRayCBIRComponent';
import Sidebar from "./components/sidebar/sidebar"
import DragDropUploader from './components/dragDropUploader/dragDropUploader';

function App() {

  const [file, setFile] = useState(null);
  const [indexActiv, setIndexActiv] = useState(0)

  {/*To be updated with similar images from backend*/}
  const similarImages = [require("./test.png"), require("./test.png")]

  function handleUpload(){
      setIndexActiv(0)
  }

  function handleShow(){
      setIndexActiv(1)
  }

  function handleFilter(){
      setIndexActiv(2)
  }
  function handleImageUploaded(file) {
      setFile(file);
      setIndexActiv(1); {/*Back to show image.  Not necessary*/}
      {/*TODO: Send image to backend*/}
  };

  return (
    <div className="App">
      <div className="container">
        <Sidebar indexActiv={indexActiv} handleUpload={handleUpload} handleShow={handleShow} handleFilter={handleFilter}/>
        <div className="others">
          {indexActiv===0 && <DragDropUploader onImageUploadedChange={handleImageUploaded}/>}
          {indexActiv===1 && file && 
            <XRayCBIR uploadedImageSource={URL.createObjectURL(file)} imgList={similarImages}/>
          }
        </div>
      </div>
    </div>
  )
}

export default App;
