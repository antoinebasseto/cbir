import { useState }  from 'react';
import {queryCache} from "../backend/BackendQueryEngine";
import { rgb } from 'd3';

export default function LatentSpaceExplorator(props) {

    const [isEditing, setIsEditing] = useState([false, false, false, false, false, false, false, false, false, false])

    function handleKeyPress (e, dim) {
        if (e.key === "Enter") {
            let temp = isEditing.map((x) => x); // We do that to copy the array
            temp[dim] = false;
            setIsEditing(temp);
        }
    };
    
    function handleEditOnClick(dim) {
        let temp = isEditing.map((x) => false); // We set everything to false
        temp[dim] = true;
        setIsEditing(temp);
    }

    function getClusterColor(dimensionNumber, index) {
        if (props.rolloutClustering.length > 0) {
            return (
                {"akiec": rgb(102, 194, 165),
                "bcc": rgb(252, 141, 98),
                "nv": rgb(141, 160, 203),
                "bkl": rgb(231, 138, 195),
                "df": rgb(166, 216, 84),
                "mel": rgb(255, 217, 47),
                "vasc": rgb(229, 196, 148),
                }[props.rolloutClustering[dimensionNumber * 11 + index]]
            )
        } else {
            return "white"
        }
    }

    return (
        <div className="flex flex-col max-h-full overflow-scroll">
            {props.dataPointFocused && !props.isGeneratingRollout &&
                props.latentSpaceImagesPath.map((arrayOflatentSpaceImagesPath, dimensionNumber) => {
                    return (
                        <div 
                            className="grid grid-cols-5 space-y-1 max-h-52 hover:bg-slate-200" 
                            id={"dim"+dimensionNumber}
                            key={"dim"+dimensionNumber}
                            onMouseEnter={() => props.handleMouseEnterRow(dimensionNumber)}
                            onMouseLeave={() => props.handleMouseLeaveRow(dimensionNumber)}
                        >
                            {isEditing[dimensionNumber] ?
                                <input 
                                    className="input_dimension" 
                                    type="text"
                                    onChange={(event) => 
                                        props.handleRenameLatent(event, dimensionNumber)
                                    } 
                                    onKeyPress={(event) => 
                                        handleKeyPress(event, dimensionNumber)
                                    }  
                                    defaultValue={props.dimensionNames[dimensionNumber]}
                                /> 
                            :
                                <div className="grid grid-cols-5 grid-rows-2 items-center">
                                    <div className="" onClick={()=> handleEditOnClick(dimensionNumber)}>
                                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} className="w-6 h-6 stroke-black hover:stroke-cyan-900">
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0115.75 21H5.25A2.25 2.25 0 013 18.75V8.25A2.25 2.25 0 015.25 6H10" />
                                        </svg>
                                    </div>
                                    <div className="max-h-8 col-span-4 text-base font-sans whitespace-nowrap overflow-scroll">{props.dimensionNames[dimensionNumber]}</div>
                                    <div className="whitespace-nowrap">weight slider</div>
                                </div>
                            }

                            <div className="col-span-4 flex justify-around">
                                {arrayOflatentSpaceImagesPath.map((latentSpaceImagePath, index) => {
                                    return (
                                        <img 
                                            className="w-1/12 h-auto border-2" 
                                            id={"image"+dimensionNumber+index} 
                                            key={"image"+dimensionNumber+index} 
                                            src={queryCache(latentSpaceImagePath)}
                                            style={{"borderColor": getClusterColor(dimensionNumber, index)}}
                                        />
                                    )
                                })}
                            </div>
                        </div>
                    )
                })
            }

            {props.isGeneratingRollout &&
                <div className="flex items-center justify-center w-full h-full">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-1/12 animate-spin">
                        <path fillRule="evenodd" d="M4.755 10.059a7.5 7.5 0 0112.548-3.364l1.903 1.903h-3.183a.75.75 0 100 1.5h4.992a.75.75 0 00.75-.75V4.356a.75.75 0 00-1.5 0v3.18l-1.9-1.9A9 9 0 003.306 9.67a.75.75 0 101.45.388zm15.408 3.352a.75.75 0 00-.919.53 7.5 7.5 0 01-12.548 3.364l-1.902-1.903h3.183a.75.75 0 000-1.5H2.984a.75.75 0 00-.75.75v4.992a.75.75 0 001.5 0v-3.18l1.9 1.9a9 9 0 0015.059-4.035.75.75 0 00-.53-.918z" clipRule="evenodd" />
                    </svg>
                </div>
            }
        </div>
    )
}
