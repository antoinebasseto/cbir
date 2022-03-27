import React from 'react';
import './App.css';
import { queryBackend } from './backend/BackendQueryEngine';
import XRayCBIR from './components/XRayCBIRComponent';

function App() {

  return (
    <div className="App">
      <XRayCBIR uploadedImageSource={"test"}/>
    </div>
  )
}

export default App;
