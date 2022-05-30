import "./similarImagesListElement.css";
import RadarChart from 'react-svg-radar-chart';
import 'react-svg-radar-chart/build/css/index.css'

export default function SimilarImagesListElement(props) {
  return (
    <li key={props.keyId} className="similarImagesListElement">
      <div className="infoContainer">
        <h2 className="label">{props.label}</h2>

        <div className = "additionalInfoContainer">
          <h5>dx_type: {props.dx_type}</h5>
          <h5>age: {props.age}</h5>
          <h5>sex: {props.sex}</h5>
          <h5>localization: {props.localization}</h5>
        </div>

        <div className = "radarChart">
          <RadarChart 
            captions={props.captions}
            data={props.data} 
            options ={props.options} 
            size={200}
          />
        </div>

        <h5 className="similarityText">Similarity: {props.similarity}</h5>

      </div>
      <img className="similarImage" src={props.imgId} />
    </li>
  )
}
