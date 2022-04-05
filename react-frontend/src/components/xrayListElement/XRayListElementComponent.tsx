import React from 'react';
import "./xrayListElement.css"

class XRayListElementComponent extends React.Component<{imgSource: string, label: string, simScore: number}, {}> {

    
    render() {
        return (
          <li className='XRayListElement'>
              <img className="similarImage" src={this.props.imgSource}/>
              <text className="diseaseText">label: {this.props.label}</text>
              <text className="similarityText">similarity: {this.props.simScore}</text>
          </li>
        );
    } 
}
export default XRayListElementComponent;

