import React, { useState }  from 'react';
import './App.css';
import { queryBackend } from './backend/BackendQueryEngine';
import XRayCBIR from './components/XRayCBIRComponent';
import Sidebar from "./components/sidebar/sidebar"
import DragDropUploader from './components/dragDropUploader/dragDropUploader';

function App() {

  const [file, setFile] = useState(null);
  const [indexActiv, setIndexActiv] = useState(0)

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
          {/*{file && <XRayCBIR uploadedImageSource={URL.createObjectURL(file)}/>}*/}
          {indexActiv===0 && <DragDropUploader onImageUploadedChange={handleImageUploaded}/>}
          {indexActiv===1 && file && 
            <div className="uploadedImageContainer">
              <img className="uploadedImage" src={URL.createObjectURL(file)}/>
            </div>
          }
        </div>
      </div>
    </div>
  )
}

export default App;
