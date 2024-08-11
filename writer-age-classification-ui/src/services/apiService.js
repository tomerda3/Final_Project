// services/apiService.js

import axios from 'axios';
import { fileToBase64 } from '../utils/fileUtils';

const API_URL = 'http://localhost:8000'; 

export const uploadHandwriting = async (file, dataSet, modelName) => {
  try {
    const base64 = await fileToBase64(file);

    const response = await axios.post(`${API_URL}/upload`, 
      {
        file: base64,
        data_set: dataSet,
        model_name: modelName
      },
      {
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );

    console.log('Response from server:', response.data);

    return response;
  } catch (error) {
    console.error('Error converting file to base64:', error);
    throw error;
  }
};

export const getClassificationResults = (id) => {
  return axios.get(`${API_URL}/results/${id}`);
};
