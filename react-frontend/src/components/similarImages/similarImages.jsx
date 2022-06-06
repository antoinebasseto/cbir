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
				{props.imgList.map((imageInfos) => {
					const [lesion_id, image_id, dx, dx_type, age, sex, localization, latent_coordinate0, latent_coordinate1, latent_coordinate2, latent_coordinate3, latent_coordinate4, latent_coordinate5, latent_coordinate6, latent_coordinate7, latent_coordinate8, latent_coordinate9, latent_coordinate10, latent_coordinate11, dist] = imageInfos
					const data = [
						{
							data: {
							        dim0: latent_coordinate0,
								dim1: latent_coordinate1,
								dim2: latent_coordinate2,
								dim3: latent_coordinate3,
								dim4: latent_coordinate4,
								dim5: latent_coordinate5,
								dim6: latent_coordinate6,
								dim7: latent_coordinate7,
								dim8: latent_coordinate8,
								dim9: latent_coordinate9,
								dim10: latent_coordinate10,
								dim11: latent_coordinate11								
							},
							// meta: { color: 'blue' }
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
								keyId={image_id} 
								imgId={queryImage(image_id)}
								label={dx} 
								dx_type={dx_type}
								age={age}
								sex={sex}
								localization={localization}
								similarity={dist} 
								data={data} 
								captions={captions} 
								options={options}
							/>
				})}
			</ol>
		</div>
    )
}
