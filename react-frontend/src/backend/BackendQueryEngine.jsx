export const BASE_URL = 'http://127.0.0.1:8000';
const PATH_GET_IMAGES = "/image/?name=";

export const queryBackend = async (route, method = "POST") => {
    const requestURL = `${BASE_URL}/${route}`;
    const data = await fetch(requestURL,
        {
            method: method
        }
    ).then(response => response.json());

    return data;
}

export const queryImage = (imageName) =>{
    const requestURL = `${BASE_URL}${PATH_GET_IMAGES}${imageName}`;
    return requestURL;
}

export default queryBackend;
