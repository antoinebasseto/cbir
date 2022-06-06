import "./sidebar.css"
import {FiUpload} from "react-icons/fi"
import {IoMdSettings} from "react-icons/io"
import {BsImages} from "react-icons/bs"
import {VscActivateBreakpoints} from "react-icons/vsc"
import {MdTravelExplore} from "react-icons/md"
import Filters from "../filters/filters"


export default function Sidebar(props) {


    return (
    <div className="sidebar">
        <div className="sidebarWrapper">
            <div className="sidebarMenu">
                <h3 className="sidebarTitle">Other Weird Moles</h3>
                <ul className="sidebarList">
                    <li className={`sidebarListItem ${props.indexActiv===0 ? "active" : ""}`} 
                        onClick={props.handleUpload}>
                        <FiUpload className="sidebarIcon"/>
                        Upload
                    </li>
                    <li className={`sidebarListItem ${props.indexActiv===1 ? "active" : ""} ${props.file ? "" : "greyed_out"}`} 
                        onClick={props.handleShow}>
                        <BsImages className="sidebarIcon"/>
                        Similar Images
                    </li>
                    <li className={`sidebarListItem ${props.indexActiv===2 ? "active" : ""}`} 
                        onClick={props.handleShowProjection}>
                        <VscActivateBreakpoints className="sidebarIcon"/>
                        Projection
                    </li>
                    <li className={`sidebarListItem ${props.indexActiv===3 ? "active" : ""} ${props.file ? "" : "greyed_out"}`} 
                        onClick={props.handleShowExplore}>
                        <MdTravelExplore className="sidebarIcon"/>
                        Explore dimensions
                    </li>
                    <li className={`sidebarListItem ${props.filterActiv ? "active" : ""} ${props.file ? "" : "greyed_out"}`} 
                        onClick={props.handleFilter}>
                        <IoMdSettings className="sidebarIcon"/>
                        Filter
                    </li>

                    {
                        props.filterActiv && 
                        <li>
                            <Filters
                                similarityThreshold={props.similarityThreshold} 
                                maxNumberImages={props.maxNumberImages} 
                                ageInterval={props.ageInterval} 
                                diseasesFilter={props.diseasesFilter} 
                                setDiseasesFilter={props.setDiseasesFilter}
                                setSimilarityThreshold={props.setSimilarityThreshold}
                                setMaxNumberImages={props.setMaxNumberImages} 
                                setAgeInterval={props.setAgeInterval} 
                                applyOnClickHandle={props.applyOnClickHandle}
                                similarityWeights={props.similarityWeights} 
                                handleFilterWeightsChange={props.handleFilterWeightsChange}
                                latentSpaceExplorationNames={props.latentSpaceExplorationNames}
                            />
                        </li>
                    }
                    
                </ul>
            </div>
        </div>
    </div>
    )
}
