import "./similarImagesListElement.css";
import RadarChart from 'react-svg-radar-chart';
import 'react-svg-radar-chart/build/css/index.css'

export default function SimilarImagesListElement(props) {
  return (
    <li key={props.keyId} className="similarImagesListElement">
      <div className="infoContainer">
        <h2 className="label">{props.label}</h2>

        <div className = "additionalInfoContainer">
          <h5>Confirmation Method: {props.dx_type}</h5>
          <h5>Age: {props.age}</h5>
          <h5>Sex: {props.sex}</h5>
          <h5>Localization: {props.localization}</h5>
        </div>

        <h5 className="similarityText">Similarity: {props.similarity}</h5>

      </div>
      <div className = "radarChart">
          <RadarChart 
            captions={props.captions}
            data={props.data} 
            options ={props.options} 
            size={200}
          />
        </div>
      <div className="imageContainer">
        <img className="similarImage" src={props.imgId}/>
      </div>
    </li>
  )
}
