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
            if (props.demo === 'skin') {
                return (
                    { //'akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'
                        "akiec": rgb(102,194,165),
                        "bcc": rgb(252,141,98),
                        "bkl": rgb(141,160,203),
                        "df": rgb(231,138,195),
                        "mel": rgb(166,216,84),
                        "nv": rgb(255,217,47),
                        "vasc": rgb(229,196,148),
                    }[props.rolloutClustering[dimensionNumber][index]]
                )
            } else if (props.demo === 'mnist') {
                return (
                    {
                        "0": rgb(141,211,199),
                        "1": rgb(255,255,179),
                        "2": rgb(190,186,218),
                        "3": rgb(251,128,114),
                        "4": rgb(128,177,211),
                        "5": rgb(253,180,98),
                        "6": rgb(179,222,105),
                        "7": rgb(252,205,229),
                        "8": rgb(217,217,217),
                        "9": rgb(188,128,189),
                    }[props.rolloutClustering[dimensionNumber][index]]
                )
            }
        } else {
            return rgb(225, 225, 225)
        }
    }

    return (
        <div className="flex flex-col max-h-full overflow-scroll">
            {props.dataPointFocused && !props.isGeneratingRollout &&
                props.latentSpaceImagesPath.map((arrayOflatentSpaceImagesPath, dimensionNumber) => {
                    return (
                        <div 
                            className="grid grid-cols-5 space-y-1 max-h-52" 
                            id={"dim"+dimensionNumber}
                            key={"dim"+dimensionNumber}
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
                                <div className="grid grid-cols-6 grid-rows-2 gap-0.5 items-center">
                                    <div className="row-span-2 self-center">
                                        {props.isGeneratingRolloutProjection ?
                                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 animate-spin">
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
                                            </svg>
                                        :
                                            <div onClick={() => props.handleClickShowBackProjection(dimensionNumber)}>
                                            {
                                            (dimensionNumber === props.dimensionExplorationProjectionFocused) ?
                                                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 cursor-pointer">
                                                    <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
                                                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                                </svg>                                          
                                            :
                                                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6 cursor-pointer">
                                                    <path strokeLinecap="round" strokeLinejoin="round" d="M3.98 8.223A10.477 10.477 0 001.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0112 4.5c4.756 0 8.773 3.162 10.065 7.498a10.523 10.523 0 01-4.293 5.774M6.228 6.228L3 3m3.228 3.228l3.65 3.65m7.894 7.894L21 21m-3.228-3.228l-3.65-3.65m0 0a3 3 0 10-4.243-4.243m4.242 4.242L9.88 9.88" />
                                                </svg>
                                            }
                                            </div>
                                        }
                                    </div>
                                    <div onClick={()=> handleEditOnClick(dimensionNumber)}>
                                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} className="w-6 h-6 stroke-black cursor-pointer">
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0115.75 21H5.25A2.25 2.25 0 013 18.75V8.25A2.25 2.25 0 015.25 6H10" />
                                        </svg>
                                    </div>
                                    <div className="max-h-8 col-span-4 text-base font-sans whitespace-nowrap overflow-scroll">{props.dimensionNames[dimensionNumber]}</div>
                                    <div className="text-xs font-sans whitespace-nowrap overflow-scroll">{props.distanceWeights[dimensionNumber]}</div>
                                    <div className="whitespace-nowrap col-span-4">
                                        <input 
                                            type="range" 
                                            value={props.distanceWeights[dimensionNumber] * 100}
                                            className='w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700'
                                            onChange={(e) => props.handleFilterWeightsChange(e.target.value/100, dimensionNumber)} 
                                        />
                                    </div>
                                </div>
                            }

                            <div className="col-span-4 flex justify-around">
                                {arrayOflatentSpaceImagesPath.map((latentSpaceImagePath, index) => {
                                    return (
                                        <img 
                                            className="w-1/12 h-auto p-0.5 border-2 border-b-4" 
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
