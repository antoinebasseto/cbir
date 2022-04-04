import React, { useState } from "react";
import { FileUploader } from "react-drag-drop-files";

const fileTypes = ["JPG", "PNG", "GIF"];

export default function DragDrop(props) {

  return (
    <div>
        <FileUploader handleChange={props.onImageUploadedChange} name="file" types={fileTypes} />
    </div>
  );
}