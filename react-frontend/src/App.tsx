import React, { useEffect, useState } from 'react';
import './App.css';
import { DataArray } from './types/DataArray';
import queryBackend from './backend/BackendQueryEngine';
import ImageChoiceComponent from './components/ImageComponent';

function App() {

  const [exampleData, setExampleData] = useState<DataArray>();
  const [dataChoice, setDataChoice] = useState<string>();
  
  
  console.log(exampleData) 
  console.log(dataChoice)

  return (
    <div className="App">
      <header className="App-header"> 
        CBIR using VAE's latent space
      </header>
      <button onClick={() =>     queryBackend(`get-data?name=` + dataChoice).then((exampleData) => {
      	setExampleData(exampleData);
    	}) } >
        Display Image
      </button>
      <div>
      <img src={`data:image/jpeg;base64,${exampleData}`} />
      </div>
    </div>
  )
}

export default App;
