import React from 'react';
import XRayListComponent from '../xrayList/XRayListComponent';
import "./xrayCBIR.css"

class XRayCBIR extends React.Component<{uploadedImageSource: string, imgList:string[]}, {}> {
    
    render() {
        return (
            <div className="XRayCBIR">
                <XRayListComponent imgList={this.props.imgList}/>
                <img className="uploadedImage" src={this.props.uploadedImageSource}/>
            </div>
        );
    } 
}
export default XRayCBIR;
