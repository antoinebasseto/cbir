import "./xrayListElement.css"

export default function XrayListElement(props) {
  return (
    <li key={props.keyId} className="XRayListElement">
        <div className="imageAndDiseaseContainer">
            <h5 className="diseaseText">{props.label}</h5>
            <img className="similarImage" src={props.imgSource}/>
        </div>
        <div className = "imageInfosContainer">
            <h5 className="patientIdText"> Lesion {props.patientId}</h5>
            <h5 className="followUpText">Label: {props.label}</h5>
            <h5 className="similarityText">Similarity: {props.similarity}</h5>
        </div>
    </li>
  )
}
