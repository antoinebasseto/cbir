import React from 'react';
import XRayListComponent from './XRayListComponent';

class XRayCBIR extends React.Component<{uploadedImageSource: string}, {}> {
    
    render() {
        return (
            <div className="XRayCBIR">
                <XRayListComponent imgList={['test', 'a']}/>
                <img src={this.props.uploadedImageSource}/>
            </div>
        );
    } 
}
export default XRayCBIR;
