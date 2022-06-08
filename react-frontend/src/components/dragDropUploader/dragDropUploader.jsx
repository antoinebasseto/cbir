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
        <h2>Welcome to Skinterpret! 👋</h2>
        This tool allows you to upload a picture of a skin lesions in order to retrieve similar ones and their correspondig diagnoses!

        <h3>How to get started</h3>
        <ol>
          <li>Upload a picture using the button above!</li>
          <li>Use the Explore dimensions menu to see what each dimension represents. Each row represents the impact on the image of varying one dimension
              while keeping the others untouched. You can rename a dimension if it represents a particular feature or you think it is important to remember it.</li>
          <li>Use the weights in the filters menu to give more or less importance to a dimension while computing the distance between two images. By default all 
            dimensions are weighted equally. You can also filter the results based on some criterion such as a given disease or age range.</li>
          <li>Look at what images are considered similar and their corresponding diagnoses under the menu Similar images.</li>
          <li>Look at the Projection menu to have an overview of the full dataset in 2D. The similar images and your uploaded image will be highlighted
            so that you can easily find them. You can have more informations about one image by putting your mouse on it in the visualization.</li>
        </ol>
        <h3>How it works underneath</h3>
        Our model uses what is called a beta variational auto-encoder. In short, this is a special neural network which is trained to first compress an image to a 
        very small latent space representation, made of only 12 dimensions, and then reconstruct the image to look as close as the initial one using this compressed 
        information only. Of course the reconstructed image won't be exactly the same as the initial one, since some information will have been lost during the compression.
        However, by forcing our network to use a small latent space, we force it to keep only the useful informations to represent the images of our dataset. This is
        why we can then use it to compute the distance between different images. Furthermore, by adding some little change to our network, we are able to force it to
        learn disentangled representations. This means in theory that each dimension in the latent space only influences one particular feature in the image. This comes 
        nevertheless at the price of having worse reconstruction, hence we cannot fully disentangle them and some dimensions might be more relevant for a domain expert than
        others. That is why this dashboard provides the option to rename some dimensions which might be representative and weight the different dimensions accordingly before
        computing the actual distance.
      </div>
    </div>

  );
}