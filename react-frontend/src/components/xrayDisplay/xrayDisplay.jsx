import React from 'react'
import "./xrayDisplay.css"
import XrayListElement from "../xrayListElement/xrayListElement"

export default function XrayDisplay(props) {

    return (
        <div className="xrayDisplayContainer">
            <div className='XRayListScroller'>
                <ol>
                    {props.imgList.map((imageInfos) => {
                        const [imgSource, patientId, followUp, label, similarity] = imageInfos
                        return  <XrayListElement keyId={imgSource} imgSource={imgSource} patientId={patientId} followUp={followUp} label={label} similarity={similarity}/>
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
