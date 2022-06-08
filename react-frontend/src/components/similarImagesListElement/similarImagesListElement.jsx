import "./similarImagesListElement.css";
import RadarChart from 'react-svg-radar-chart';
import 'react-svg-radar-chart/build/css/index.css'

export default function SimilarImagesListElement(props) {
  return (
    <li key={props.keyId} className="similarImagesListElement">
      <h2 className="label">{props.label}</h2>
      <div className="bodyContainer">
        <div className="leftColumn">

          <div className="patientInfoContainer">
            <div className="patientInfoRow">
              <div className="patientInfoTitleContainer">
                <h3 className="patientInfoTitle">Confirmation method:</h3>
              </div>
              <div className="patientInfoValueContainer">
                <h3 className="patientInfoValueText">{props.dx_type}</h3>
              </div>
            </div>
            <div className="horizontalLine"></div>

            <div className="patientInfoRow">
              <h3 className="patientInfoTitle">Localization:</h3>
              <div className="patientInfoValueContainer">
                <h3 className="patientInfoValueText">{props.localization}</h3>
              </div>
            </div>
            <div className="horizontalLine"></div>

            <div className="patientInfoRow">
              <h3 className="patientInfoTitle">Age:</h3>
              <div className="patientInfoValueContainer">
                <h3 className="patientInfoValueText">{props.age}</h3>
              </div>
            </div>
            <div className="horizontalLine"></div>

            <div className="patientInfoRow">
              <h3 className="patientInfoTitle">Sex:</h3>
              <div className="patientInfoValueContainer">
                <h3 className="patientInfoValueText">{props.sex}</h3>
              </div>
            </div>
            <div className="horizontalLine"></div>
          </div>

          <div className="imageContainer">
            <img className="similarImage" src={props.imgId}/>
          </div>

        </div>
        <div className="verticalLine"></div>
        <div className="rightColumn">

          <div className = "radarChartContainer">
            <h3 className="radarTitle">Distance along each axis</h3>
            <RadarChart 
              className = "radarChart"
              captions={props.captions}
              data={props.data} 
              options ={props.options} 
              size={315}
            />
          </div>

          <div className="horizontalLine"></div>

          <div className="totalDistanceContainer">
              <h3 className="patientInfoTitle">Total distance:</h3>
              &emsp;<h3 className="boldText">{Math.round(props.distance * 1000) / 1000}</h3>
          </div>

        </div>

      </div>
    </li>
  )
}
