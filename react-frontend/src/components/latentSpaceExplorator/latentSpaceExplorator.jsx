import React from 'react'
import "./latentSpaceExplorator.css"

export default function LatentSpaceExplorator(props) {
    const BASE_BACKEND_URL = "http://127.0.0.1:8000" /* URL of the backend server */
    const PATH_GET_IMAGES = "/get_image/?imagePath=" /* Name of the method and query parameter to get the image (see app.py in backend folder)*/

    return (
        <div className="latentSpaceExploratorContainer">
            { props.uploadedImage && 
                props.latentSpaceImagesPath.map((latentSpaceImagePath) => {
                    return <img className="similarImage" src={BASE_BACKEND_URL + PATH_GET_IMAGES + latentSpaceImagePath}/>
                })
            }
        </div>
    )
}
