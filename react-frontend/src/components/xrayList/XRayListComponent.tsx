import React from 'react';
import XRayListElementComponent from '../xrayListElement/XRayListElementComponent';
import "./xrayList.css"


class XRayListComponent extends React.Component<{imgList: string[]}, {}> {

    
    render() {
      const rows: JSX.Element[] = [];
    
      this.props.imgList.forEach((imgSource) => {
        rows.push(
          <XRayListElementComponent
            imgSource={imgSource}
            key={imgSource}
            label='this is a label'
            simScore={0} />
        );
      });

        return (
          <ol className='XRayList'>
            {rows}
          </ol>
        );
    } 
}
export default XRayListComponent;
