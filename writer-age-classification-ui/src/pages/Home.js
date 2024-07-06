import React from 'react';
import { Link } from 'react-router-dom';

const Home = () => {
  return (
    <div>
      <h1>Welcome to Writer Age Classification</h1>
      <Link to="/upload">Upload Handwriting</Link>
    </div>
  );
};

export default Home;
