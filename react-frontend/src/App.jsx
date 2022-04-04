import React, { useState }  from 'react';
import './App.css';
import { queryBackend } from './backend/BackendQueryEngine';
import XRayCBIR from './components/XRayCBIRComponent';
import Sidebar from "./components/sidebar/sidebar"
import DragDrop from './components/sidebar/DragDropUploader/DragDropUploader';

function App() {

  const [file, setFile] = useState(null);

  function handleImageUploaded(file) {
      setFile(file);
      console.log(file)
  };

  return (
    <div className="App">
      <div className="container">
        <Sidebar/>
        <div className="others">
        {/*{file && <XRayCBIR uploadedImageSource={URL.createObjectURL(file)}/>}*/}
          <DragDrop onImageUploadedChange={handleImageUploaded}/>
          {file && <div className="uploadedImageContainer"><img className="uploadedImage" src={URL.createObjectURL(file)}/></div>}
        </div>
      </div>
    </div>
  )
}

export default App;
