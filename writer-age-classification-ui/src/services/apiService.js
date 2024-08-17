import axios from 'axios';
import { fileToBase64 } from '../utils/fileUtils';

const API_URL = 'http://localhost:8080'; 

export const uploadHandwriting = async (file, dataSet, modelName = 'ConvNeXtXLarge') => {
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
    console.log('Response from server:', result);
    
    return result;
  } catch (error) {
    console.error('Error converting file to base64:', error);
    throw error;
  }
};

export const getClassificationResults = (id) => {
  return axios.get(`${API_URL}/results/${id}`);
};
