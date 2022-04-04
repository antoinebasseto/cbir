import React from 'react';
import './App.css';
import { queryBackend } from './backend/BackendQueryEngine';
import XRayCBIR from './components/XRayCBIRComponent';
import Sidebar from "./components/sidebar/sidebar"

function App() {

  return (
    <div className="App">
      <div className="container">
        <Sidebar/>
        <div className="others">
          {/*<XRayCBIR uploadedImageSource={require("./test.png")}/>*/}
        </div>
      </div>
    </div>
  )
}

export default App;
