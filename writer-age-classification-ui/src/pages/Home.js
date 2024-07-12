import React from 'react';
import { Link } from 'react-router-dom';
import './Home.css';
import handwritingVideo from '../assets/handwriting.mp4';

const Home = () => {
  return (
    <div className="home-container">
      <video autoPlay loop muted className="background-video">
        <source src={handwritingVideo} type="video/mp4" />
      </video>
      <div className="content">
        <h1>Welcome to Writer Age Classification</h1>
        <div className="cards-container">
          <div className="card">
            <Link to="/models">
              <h2>Models</h2>
              <p>Learn about the models we used.</p>
            </Link>
          </div>
          <div className="card">
            <Link to="/article">
              <h2>Article</h2>
              <p>Read the detailed project article.</p>
            </Link>
          </div>
          <div className="card">
            <Link to="/about">
              <h2>About</h2>
              <p>About the project and team.</p>
            </Link>
          </div>
        </div>
        <Link to="/upload" className="start-button">Start</Link>
      </div>
    </div>
  );
};

export default Home;
