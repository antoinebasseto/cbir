import React, { useState } from "react";
import { FileUploader } from "react-drag-drop-files";
import "./dragDropUploader.css"
import {FiUpload} from "react-icons/fi"

const fileTypes = ["JPG", "PNG", "GIF"];

const children = 
  <div className="uploaderContainer">
      <FiUpload className="uploaderIcon"/>
      <h5 className="text">Upload or drop a file here</h5>
  </div>

export default function DragDropUploader(props) {

  return (
    <FileUploader handleChange={props.onImageUploadedChange} name="file" types={fileTypes} children={children}/>
  );
}