export const BASE_URL = 'http://127.0.0.1:8000';

export const queryBackend = async (route, method) => {
    const requestURL = `${BASE_URL}/${route}`;
    const data = await fetch(requestURL,
        {
            method: method
        }
    ).then(response => response.json());

    return data;
}

export const queryImages = (route) =>{
    const requestURL = `${BASE_URL}/image/?name=${route}`;
    // const dataset = await fetch(requestURL,
    //     {
    //         method: 'GET'
    //     }
    // );
    return requestURL;
}

// //added to implement images
// export const queryBackend2 = async (route: string): Promise<any> => {
//     const requestURL = `${BASE_URL}/${route}`;
//     const data = await fetch(requestURL,
//         {
//             method: 'POST'
//         }
//     ).then(response => response.json());

//     return data;
// }

export default queryBackend;
