import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { getClassificationResults } from '../services/apiService';

const Results = () => {
  const { id } = useParams();
  const [results, setResults] = useState(null);

  useEffect(() => {
    const fetchResults = async () => {
      try {
        const response = await getClassificationResults(id);
        setResults(response.data);
      } catch (error) {
        console.error('Error fetching results:', error);
      }
    };

    fetchResults();
  }, [id]);

  return (
    <div>
      <h1>Results</h1>
      {results ? (
        <div>
          <p>Age: {results.age}</p>
          <p>Gender: {results.gender}</p>
        </div>
      ) : (
        <p>Loading results...</p>
      )}
    </div>
  );
};

export default Results;
