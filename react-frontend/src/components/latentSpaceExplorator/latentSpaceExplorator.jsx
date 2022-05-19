import "./latentSpaceExplorator.css"
import {queryImage} from "../../backend/BackendQueryEngine";

export default function LatentSpaceExplorator(props) {

    return (
        <div className="latentSpaceExploratorContainer">
            {props.uploadedImage && 
                props.latentSpaceImagesPath.map((arrayOflatentSpaceImagesPath, dimensionNumber) => {
                    return (
                        <div className = "rollout_full_row_container">
                            <div className="dim_text">Dimension {dimensionNumber+1}</div>
                            <div className = "rollout_row_images_container">
                                {arrayOflatentSpaceImagesPath.map((latentSpaceImagePath) => {
                                    return <img className="rollout_image" src={queryImage(latentSpaceImagePath)}/>
                                })}
                             </div>

                        </div>
                    )
                })
            }
        </div>
    )
}
