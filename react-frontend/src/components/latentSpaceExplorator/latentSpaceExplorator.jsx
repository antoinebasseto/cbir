import { useState}  from 'react';
import "./latentSpaceExplorator.css"
import {queryImage} from "../../backend/BackendQueryEngine";
import {BiEditAlt} from "react-icons/bi"
import {queryCache} from "../../backend/BackendQueryEngine";

export default function LatentSpaceExplorator(props) {

    const [isEditing, setIsEditing] = useState([false, false, false, false, false, false, false, false, false, false])

    function handleKeyPress (e, dim) {
        if (e.key === "Enter") {
            let temp = isEditing.map((x) => x); //We do that to copy the array
            temp[dim] = false;
            setIsEditing(temp);
        }
      };
    
      function handleEditOnClick(dim){
        let temp = isEditing.map((x) => x); //We do that to copy the array
        temp[dim] = true;
        setIsEditing(temp);
      }

    return (
        <div className="latentSpaceExploratorContainer">
            {props.uploadedImage && 
                props.latentSpaceImagesPath.map((arrayOflatentSpaceImagesPath, dimensionNumber) => {
                    return (
                        <div className = "rollout_full_row_container">
                            {isEditing[dimensionNumber] ?
                            <input className = "input_dimension" type = 'text' onChange={(event) => props.handleRenameLatent(event, dimensionNumber)} onKeyPress={(event) => handleKeyPress(event, dimensionNumber)}  defaultValue = {props.dimensionNames[dimensionNumber]}/> 
                            :
                            <div className = "container_dim_text_and_editor">
                                <div className="dim_text">{props.dimensionNames[dimensionNumber]}</div>
                                <div className="edit_button" onClick ={()=> handleEditOnClick(dimensionNumber)}>
                                    <BiEditAlt className="edit_icon"/>
                                </div>
                            </div>
                            }
                            <div className = "rollout_row_images_container">
                                {arrayOflatentSpaceImagesPath.map((latentSpaceImagePath) => {
                                    return <img className="rollout_image" src={queryCache(latentSpaceImagePath)}/>
                                })}
                             </div>

                        </div>
                    )
                })
            }
        </div>
    )
}
