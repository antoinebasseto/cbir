import "./xrayDisplay.css"
import {queryImage} from "../../backend/BackendQueryEngine";
import XrayListElement from "../xrayListElement/xrayListElement"

export default function XrayDisplay(props) {

    return (
        <div className="xrayDisplayContainer">
            <div className='XRayListScroller'>
                <ol>
                    {props.imgList.map((imageInfos) => {
                        const [imgSource, patientId, followUp, label, similarity] = imageInfos
                        return  <XrayListElement keyId={imgSource} imgSource={queryImage(imgSource)} patientId={patientId} followUp={followUp} label={label} similarity={similarity}/>
                    })}
                </ol>
            </div>
            <div className="verticalLine"></div>
            <div className="imageContainer">
                <img className="uploadedImage" src={props.uploadedImageSource}/>
            </div>
        </div>
    )
}
