import axios from 'axios';
import { fileToBase64 } from '../utils/fileUtils';

const API_URL = 'http://localhost:8080'; 

export const uploadHandwriting = async (file, dataSet, modelName) => {
  try {
    const base64 = await fileToBase64(file);

    const response = await axios.post(`${API_URL}/get_prediction`, 
      {
        image: base64,
        data_set: dataSet,
        model_name: modelName
      },
    );
    const result = response.data.predictions[0]
    if (modelName === 'ConvNeXtXLargeRegression'){
      return result;
    }
    let range;
    if (result === 1) {
      range = '0-15';
    } else if (result === 2) {
      range = '16-25';
    } else if (result === 3) {
      range = '26-50';
    } else if (result === 4) {
      range = '51+';
    } else {
      range = String(result);
    }
    console.log('Response from server:', range);
    return range;
    
    
  } catch (error) {
    console.error('Error converting file to base64:', error);
    throw error;
  }
};

export const getClassificationResults = (id) => {
  return axios.get(`${API_URL}/results/${id}`);
};
