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
								dim0: img["latent_coordinate_0"],
								dim1: img["latent_coordinate_1"],
								dim2: img["latent_coordinate_2"],
								dim3: img["latent_coordinate_3"],
								dim4: img["latent_coordinate_4"],
								dim5: img["latent_coordinate_5"],
								dim6: img["latent_coordinate_6"],
								dim7: img["latent_coordinate_7"],
								dim8: img["latent_coordinate_8"],
								dim9: img["latent_coordinate_9"],
								dim10: img["latent_coordinate_10"],
								dim11: img["latent_coordinate_11"]								
							},
							meta: { color: 'blue' }
						}
					];

					const captions = {
						    dim0: "Latent Coordinate 1",
							dim1: "Latent Coordinate 2",
							dim2: "Latent Coordinate 3",
							dim3: "Latent Coordinate 4",
							dim4: "Latent Coordinate 5",
							dim5: "Latent Coordinate 6",
							dim6: "Latent Coordinate 7",
							dim7: "Latent Coordinate 8",
							dim8: "Latent Coordinate 9",
							dim9: "Latent Coordinate 10",
							dim10: "Latent Coordinate 11",
							dim11: "Latent Coordinate 12"	
					};

					const options = {
						captionProps: () => ({
							className: 'caption',
							textAnchor: 'middle',
							fontSize: 13,
							fontFamily: 'sans-serif'
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
								similarity={img["dist"]} 
								data={data} 
								captions={captions} 
								options={options}
							/>
				})}
			</ol>
		</div>
    )
}
