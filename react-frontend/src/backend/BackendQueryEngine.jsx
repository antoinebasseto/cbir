export const BASE_URL = 'http://127.0.0.1:8000';
const PATH_GET_IMAGES = "/image?name=";

export const queryBackend = async (route, method) => {
    const requestURL = `${BASE_URL}/${route}`;
    const data = await fetch(requestURL,
        {
            method: method
        }
    ).then(response => response.json());

    return data;
}

export const queryBackendWithFile = async (route, file) => {
    const requestURL = `${BASE_URL}/${route}`;
    
    let fr = new FormData();
    //data.append('filters', 'filtersPlaceholder');
    fr.append('file', file, file.name);
    const data = await fetch(requestURL,
        {
            method: "POST",
            body: fr}
    ).then(response => response.json());

    return data;
}


export const updateFiltersBackend = async(route, method='POST', distanceWeights, maxNumberImages, ageInterval, diseasesFilter) => {
    const requestURL = `${BASE_URL}/${route}`;
    const data = await fetch(requestURL,
        {
            method: method,
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({distanceWeights: distanceWeights,
                                  maxNumberImages: maxNumberImages, 
                                  ageInterval: ageInterval, 
                                  diseasesFilter: diseasesFilter,
                                })
        }
    ).then(response => response.json());

    return data;
}

export const queryImage = (imageName) =>{
    const requestURL = `${BASE_URL}${PATH_GET_IMAGES}${imageName}`;
    return requestURL;
}

export const queryCache = (imageName) =>{
    const requestURL = `${BASE_URL}/cache/?name=${imageName}`;
    return requestURL;
}

export default queryBackend;
