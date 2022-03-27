import React from 'react';

class XRayListElementComponent extends React.Component<{imgSource: string, label: string, simScore: number}, {}> {

    
    render() {
        return (
          <li className='XRayListElement'>
              <img src={this.props.imgSource}/>
              <text>label: {this.props.label}</text>
              <text>similarity: {this.props.simScore}</text>
          </li>
        );
    } 
}
export default XRayListElementComponent;

