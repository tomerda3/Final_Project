import axios from 'axios';

const API_URL = 'http://localhost:8000'; // Replace with your backend URL

export const uploadHandwriting = (file) => {
  const formData = new FormData();
  formData.append('file', file);

  return axios.post(`${API_URL}/upload`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};

export const getClassificationResults = (id) => {
  return axios.get(`${API_URL}/results/${id}`);
};
