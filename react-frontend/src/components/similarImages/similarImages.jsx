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
					const [lesion_id, image_id, dx, dx_type, age, sex, localization, latent_coordinate1, latent_coordinate2, latent_coordinate3, latent_coordinate4, latent_coordinate5, latent_coordinate6, latent_coordinate7, latent_coordinate8, latent_coordinate9, latent_coordinate10, latent_coordinate11, dist] = imageInfos
					const data = [
						{
							data: {
								brightness: latent_coordinate1,
								redness: latent_coordinate2,
								size: latent_coordinate3,
								opacity: latent_coordinate4,
								fuzziness: latent_coordinate5
							},
							// meta: { color: 'blue' }
						}
					];

					const captions = {
						// columns
						brightness: "Brightness",
						redness: "Redness",
						size: "Size",
						opacity: "Opacity",
						fuzziness: "Fuzziness",
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
