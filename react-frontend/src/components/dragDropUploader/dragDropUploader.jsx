import React, { useState } from "react";
import { FileUploader } from "react-drag-drop-files";
import "./dragDropUploader.css"
import {FiUpload} from "react-icons/fi"

const fileTypes = ["JPG", "PNG", "GIF"];

const children = 
  <div className="uploaderContainer">
      <FiUpload className="uploaderIcon"/>
      <h5 className="text">Drag or click to upload a file</h5>
  </div>

export default function DragDropUploader(props) {

  return (
    <div className="uploaderScreenContainer">
      <FileUploader handleChange={props.onImageUploadedChange} name="file" types={fileTypes} children={children}/>
      <div className="textContainer">
        <h2>Welcome to Other Weird Moles! 👋</h2>
        This tool allows you to upload a picture of a skin lesions in order to retrieve similar ones and their correspondig diagnoses!
        <div className="columnsContainer">
          <div className="firstColumn">
            <h3>How to get started</h3>
            <ol>
              <li>Upload a picture using the button above!</li>
              <li>Look at what images are considered similar under the menu....</li>
            </ol>
            <h3>How it works</h3>
          </div>
          <div className="secondColumn">
            <h3>Who we are</h3>
          </div>
        </div>
      </div>
    </div>
  );
}