import "./similarImagesListElement.css";
import RadarChart from 'react-svg-radar-chart';
import 'react-svg-radar-chart/build/css/index.css'

export default function SimilarImagesListElement(props) {
  return (
    <li key={props.keyId} className="similarImagesListElement">
      <h2 className="label">{props.label}</h2>
      <div className="infoAndGraphicsContainer">
        <div className="infoContainer">
          <p className = "additionalInfoContainer">
            <b>Confirmation method</b><br/>
            &emsp;{props.dx_type}<br/>
            <b>Localization</b><br/>
            &emsp;{props.localization}<br/>
            <b>Age</b><br/>
            &emsp;{props.age}<br/>
            <b>Sex</b><br/>
            &emsp;{props.sex}<br/>
            <b className="similarityText">Distance</b><br/>
            &emsp;{Math.round(props.distance * 1000) / 1000}
          </p>
        </div>
        <div className = "radarChartContainer">
          <p>Distance to uploaded image<br/>along each dimension</p>
          <RadarChart 
            className = "radarChart"
            style="padding: 10px !important;"
            captions={props.captions}
            data={props.data} 
            options ={props.options} 
            size={200}
          />
          </div>
        <div className="imageContainer">
          <img className="similarImage" src={props.imgId}/>
        </div>
      </div>
    </li>
  )
}
