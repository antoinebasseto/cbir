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
								dim0: img["latent_distance_0"],
								dim1: img["latent_distance_1"],
								dim2: img["latent_distance_2"],
								dim3: img["latent_distance_3"],
								dim4: img["latent_distance_4"],
								dim5: img["latent_distance_5"],
								dim6: img["latent_distance_6"],
								dim7: img["latent_distance_7"],
								dim8: img["latent_distance_8"],
								dim9: img["latent_distance_9"],
								dim10: img["latent_distance_10"],
								dim11: img["latent_distance_11"]
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
								key={img["image_id"]}
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
