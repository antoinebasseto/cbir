import "./similarImages.css"
import {queryImage} from "../../backend/BackendQueryEngine";
import SimilarImagesListElement from "../similarImagesListElement/similarImagesListElement"

export default function SimilarImages(props) {
	
    return (
        <div className="similarImagesContainer">
			<div className="imageContainer">
                <img className="uploadedImage" src={props.uploadedImageSource}/>
            </div>
			<ol className="similarImagesListScroller">
				{props.imgList.map((img) => {
					const data = [
						{
							data: {
								dim0: Math.abs(img["latent_coordinate_0"] - props.latentSpace[0]),
								dim1: Math.abs(img["latent_coordinate_1"] - props.latentSpace[1]),
								dim2: Math.abs(img["latent_coordinate_2"] - props.latentSpace[2]),
								dim3: Math.abs(img["latent_coordinate_3"] - props.latentSpace[3]),
								dim4: Math.abs(img["latent_coordinate_4"] - props.latentSpace[4]),
								dim5: Math.abs(img["latent_coordinate_5"] - props.latentSpace[5]),
								dim6: Math.abs(img["latent_coordinate_6"] - props.latentSpace[6]),
								dim7: Math.abs(img["latent_coordinate_7"] - props.latentSpace[7]),
								dim8: Math.abs(img["latent_coordinate_8"] - props.latentSpace[8]),
								dim9: Math.abs(img["latent_coordinate_9"] - props.latentSpace[9]),
								dim10: Math.abs(img["latent_coordinate_10"] - props.latentSpace[10]),
								dim11: Math.abs(img["latent_coordinate_11"] - props.latentSpace[11])
							},
							meta: { color: "#104242" }
						}
					];

					var captions = {}
					props.dimensionNames.forEach((el, index) => captions["dim" + index] = el)

					const options = {
						captionMargin: 50,
						captionProps: () => ({
							className: 'caption',
							textAnchor: 'middle',
							fontSize: 13,
							fontFamily: 'sans-serif',
						})
					};

					return  <SimilarImagesListElement 
								keyId={img["image_id"]}
								imgId={queryImage(img["image_id"])}
								label={img["dx"]} 
								dx_type={img["dx_type"]}
								age={img["age"]}
								sex={img["sex"]}
								localization={img["localization"]}
								distance={img["dist"]} 
								data={data} 
								captions={captions} 
								options={options}
							/>
				})}
			</ol>
		</div>
    )
}
