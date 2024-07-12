import React from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  CardMedia,
  Typography,
  Box,
  Divider,
  Button,
} from '@mui/material';
import { styled } from '@mui/system';
import { useNavigate } from 'react-router-dom';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';

import maayanImage from '../assets/maayan.jpg';
import tomerImage from '../assets/tomer.jpg';
import ofriImage from '../assets/ofri.jpg';
import irinaImage from '../assets/irina.jpg';
import marinaImage from '../assets/marina.jpg';

const participants = [
  {
    name: 'Ofri Rom',
    description: 'Ofri Rom plays a vital role in the project, with a focus on model optimization and evaluation.',
    image: ofriImage,
  },
  {
    name: 'Tomer Damti',
    description: 'Tomer Damti is a key participant in the project, contributing expertise in data science and deep learning.',
    image: tomerImage,
  },
  {
    name: 'Maayan Rabinovitch',
    description: 'Maayan Rabinovitch is a dedicated participant in the Writer Age Classification project, bringing strong analytical skills and a passion for machine learning.',
    image: maayanImage,
  },
];

const advisors = [
  {
    name: 'Dr. Irina Rabaev',
    description: 'Irina Rabaev received her Ph.D. in Computer Science from Ben-Gurion University of the Negev, Israel. She is currently a faculty member at Department of Software Engineering of Shamoon Academic College of Engineering, Beer-Sheva, Israel. Her main research interests include the areas of computer vision and image processing with a focus on historical documents analysis.',
    image: irinaImage,
  },
  {
    name: 'Dr. Marina Litvak',
    description: 'Marina Litvak has obtained a Ph.D. in Information Systems Engineering from Ben-Gurion University of the Negev in 2010. She is currently a faculty member at Department of Software Engineering of Shamoon Academic College of Engineering in Beer Sheva, Israel. Her research interests include information retrieval, data mining, text mining, text analysis, recommender systems, and social media.',
    image: marinaImage,
  },
];

const FloatingCard = styled(Card)({
  transition: 'transform 0.3s, box-shadow 0.3s',
  '&:hover': {
    transform: 'translateY(-10px)',
    boxShadow: '0 8px 16px rgba(0, 0, 0, 0.2)',
  },
});

const About = () => {
  const navigate = useNavigate();

  return (
    <Container sx={{ py: 3 }}>
      <Button
        variant="contained"
        startIcon={<ArrowBackIcon />}
        onClick={() => navigate(-1)}
        sx={{ mb: 2 }}
      >
        Back
      </Button>
      <Typography variant="h3" align="center" gutterBottom>
        About
        <Divider />
      </Typography>
      <Box my={4} />
      <Typography variant="h4" align="center" gutterBottom>
        Academic Advisors
      </Typography>
      <Grid container spacing={4}>
        {advisors.map((advisor, index) => (
          <Grid item key={index} xs={12} sm={6} md={4}>
            <FloatingCard>
              {advisor.image && (
                <CardMedia
                  component="img"
                  height="200"
                  image={advisor.image}
                  alt={advisor.name}
                />
              )}
              <CardContent>
                <Typography gutterBottom variant="h5" component="div">
                  {advisor.name}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  {advisor.description}
                </Typography>
              </CardContent>
            </FloatingCard>
          </Grid>
        ))}
      </Grid>
      <Box my={4} />
      <Typography variant="h4" align="center" gutterBottom>
        Participants
      </Typography>
      <Grid container spacing={4}>
        {participants.map((participant, index) => (
          <Grid item key={index} xs={12} sm={6} md={4}>
            <FloatingCard>
              {participant.image && (
                <CardMedia
                  component="img"
                  height="200"
                  image={participant.image}
                  alt={participant.name}
                />
              )}
              <CardContent>
                <Typography gutterBottom variant="h5" component="div">
                  {participant.name}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  {participant.description}
                </Typography>
              </CardContent>
            </FloatingCard>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
};

export default About;
