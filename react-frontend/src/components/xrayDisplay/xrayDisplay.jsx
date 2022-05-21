import "./xrayDisplay.css"
import {queryImage} from "../../backend/BackendQueryEngine";
import XrayListElement from "../xrayListElement/xrayListElement"

export default function XrayDisplay(props) {

    return (
        <div className="xrayDisplayContainer">
            <div className='XRayListScroller'>
                <ol>
                    {props.imgList.map((imageInfos) => {
                        const [imgSource, patientId, followUp, label, similarity, dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8, dim9, dim10, dim11, dim12] = imageInfos
			const data = [
			      {
				data: {
				  brightness: dim1,
				  redness: dim2,
				  size: dim3,
				  opacity: dim4,
				  fuzziness: dim5
				},
				meta: { color: 'blue' }
			      }
			    ];

			const captions = {
			      // columns
			      brightness: 'Brightness',
			      redness: 'Redness',
			      size: 'Size',
			      opacity: "Opacity",
			      fuzziness: "Fuzziness",
			    };
			const options = {
			  captionProps: () => ({
			    className: 'caption',
			    textAnchor: 'middle',
			    fontSize: 13,
			    fontFamily: 'sans-serif'
			  }),

			};
                        return  <XrayListElement keyId={imgSource} imgSource={queryImage(imgSource)} patientId={patientId} followUp={followUp} label={label} similarity={similarity} data={data} 					captions= {captions} options = {options}/>
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
