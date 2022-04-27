export interface queryBackendProps {
    route: string;
}

export const BASE_URL = 'http://127.0.0.1:8000';

export const queryBackend = async (route: string): Promise<any> => {
    const requestURL = `${BASE_URL}/${route}`;
    const data = await fetch(requestURL,
        {
            method: 'POST'
        }
    ).then(response => response.json());

    return data;
}
//added to implement images
export const queryBackend2 = async (route: string): Promise<any> => {
    const requestURL = `${BASE_URL}/${route}`;
    const data = await fetch(requestURL,
        {
            method: 'POST'
        }
    ).then(response => response.json());

    return data;
}

export const queryImages = (route: string) =>{
    const requestURL = `${BASE_URL}/image/?name=${route}`;
    // const data = await fetch(requestURL,
    //     {
    //         method: 'GET'
    //     }
    // );
    return requestURL;
}


export default queryBackend;
