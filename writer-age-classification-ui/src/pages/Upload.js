import React, { useState } from 'react';
import { Container, Box, Button, CircularProgress, Grid, Stack, styled, Divider, RadioGroup, FormControlLabel, Radio, Card, CardContent, Typography, CardMedia } from '@mui/material';
import { uploadHandwriting } from '../services/apiService';
import video1 from '../assets/video1.mp4';
import video2 from '../assets/video2.mp4';
import video3 from '../assets/video3.mp4';
import convNeXtXLargeImage from '../assets/ConvNeXtXLargeimage.jpg';
import convNeXtXLargeRegressionImage from '../assets/convNeXtXLargeRegressionImage.jpg'
import backGroundVideo from '../assets/net.mp4';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import { useNavigate } from 'react-router-dom';
const FloatingCard = styled(Card)({
  transition: 'transform 0.3s, box-shadow 0.3s',
  '&:hover': {
    transform: 'translateY(-10px)',
    boxShadow: '0 8px 16px rgba(0, 0, 0, 0.2)',
  },
});
const BlueContainer = styled(Container)({
  backgroundColor: '#f0f8ff', 
  borderRadius: '8px',
  padding: '2rem',
  textAlign: 'center',
  minHeight: '100vh', 
  display: 'flex',
  flexDirection: 'column',
  justifyContent: 'space-between',
});

const UploadButton = styled(Button)({
  backgroundColor: '#1e90ff', 
  color: '#fff',
  '&:hover': {
    backgroundColor: '#4682b4',
  },
});

const VideoContainer = styled(Box)({
  display: 'flex',
  justifyContent: 'center',
  gap: '1rem',
  marginTop: '2rem',
});

const Video = styled('video')({
  width: '300px',
  height: '200px',
  borderRadius: '8px',
  boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.1)',
});

const modelData = [
  {
    name: 'ConvNeXtXLarge',
    description: 'ConvNeXtXLarge is a high-capacity deep convolutional neural network designed with a modernized architecture inspired by ResNet. It excels in capturing intricate patterns and features within handwriting, making it highly effective for age classification tasks.',
    parameters: 'Over 350 million',
    layers: 'Convolutional layers with batch normalization and ReLU activation, followed by fully connected layers for classification',
    image: convNeXtXLargeImage, 
  },
  {
    name: 'ConvNeXtXLargeRegression',
    description: 'ConvNeXtXLargeRegression is a variation of ConvNeXtXLarge tailored for regression tasks. Instead of classifying age into discrete categories, it predicts the age as a continuous variable, allowing for more precise age estimation based on handwriting analysis.',
    parameters: 'Over 350 million',
    layers: 'Convolutional layers with batch normalization and ReLU activation, followed by regression layers to output a continuous age value',
    image: convNeXtXLargeRegressionImage,
  },
  {
    name: 'Transformer',
    description: 'Transformers are a type of neural network architecture known for their ability to handle sequential data. In the context of handwriting analysis, Transformers excel at capturing temporal dependencies and the sequential nature of handwriting strokes, making them suitable for age classification tasks.',
    parameters: 'Varies depending on implementation, typically hundreds of millions',
    layers: 'Self-attention layers followed by feedforward neural networks, capable of capturing long-range dependencies in data',
    // image: transformerImage,
  },
];

const Upload = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const navigate = useNavigate();
  const [selectedModel, setSelectedModel] = useState('ConvNeXtXLarge'); 
  const handleSelectionChange = (model) => {
    setSelectedModel(model);
    console.log('Selected Model:', model.name);
  };
  const [dataSet, setDataSet] = useState('HHD'); 
  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (file) {
      setLoading(true);
      try {
        console.log(selectedModel);
        const response = await uploadHandwriting(file, dataSet, selectedModel.name);
        setResult(response); 
      } catch (error) {
        console.error('Error uploading file:', error);
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <BlueContainer>
      <video autoPlay loop muted className="background-video">
        <source src={backGroundVideo} type="video/mp4" />
      </video>
      <Box mb={4}>
        <Typography variant="h3" gutterBottom>
          Welcome to the Handwriting Analysis System
        </Typography>
        <Divider sx={{ my: 1, borderBottomWidth: 2, borderColor: '#1e90ff' }} />
        <Typography variant="h6" paragraph>
          To get started, please upload a picture of your handwriting. Our system will process the image and provide you with the analysis results.
        </Typography>
      </Box>
      <Box>
        <Typography variant="h5" gutterBottom>
          Upload Your Handwriting Picture
        </Typography>
          <input type="file" onChange={handleFileChange} />
        <Divider sx={{ my: 1, borderBottomWidth: 2, borderColor: '#1e90ff' }} />
        <Typography variant="h5" gutterBottom>
          Choose a model
        </Typography>
        <Grid container spacing={4}>
            {modelData.map((model, index) => (
              <Grid item key={index} xs={12} sm={6} md={4}>
                <FloatingCard onClick={() => handleSelectionChange(model)}
                    sx={{
                      border: selectedModel?.name === model.name ? '2px solid #1976d2' : 'none',
                      boxShadow: selectedModel?.name === model.name ? '0 4px 20px rgba(25, 118, 210, 0.3)' : 'none',
                    }}
                  >
                  <CardContent>
                    <Typography gutterBottom variant="h5" component="div">
                      {model.name}
                    </Typography>
                    {model.image && (
                    <CardMedia
                      component="img"
                      height="200"
                      image={model.image}
                      alt={model.name}                      
                    />
                  )}
                    <Typography variant="body2" color="textSecondary">
                      {model.description}
                    </Typography>
                    <Box mt={2}>
                      <Typography variant="body2">
                        <strong>Parameters:</strong> {model.parameters}
                      </Typography>
                      <Typography variant="body2">
                        <strong>Layers:</strong> {model.layers}
                      </Typography>
                    </Box>
                  </CardContent>
                </FloatingCard>
              </Grid>
            ))}
          </Grid>
      </Box>
      <form onSubmit={handleSubmit}>
            <UploadButton type="submit" variant="contained">
              {loading ? <CircularProgress size={24} color="inherit" /> : 'PREDICT AGE'}
            </UploadButton>
      </form>
      {result && (
          <Box mt={4} p={15} border={2} borderColor="#1e90ff" borderRadius="8px" bgcolor="#e6f7ff">
            <Typography variant="h6" gutterBottom color="#1e90ff">
              Estimated Age Based on Handwriting
            </Typography>
            <Box display="flex" flexDirection="column" alignItems="center">
              <Typography variant="h4" color="#333" fontWeight="bold">
                {result} years
              </Typography>
              <Typography variant="body1" color="#555" mt={1}>
                Our system has analyzed the handwriting and estimated the writer's age based on the provided image.
              </Typography>
            </Box>
          </Box>
        )}
      <Box mt="auto">
        <VideoContainer>
          <Video autoPlay loop muted>
            <source src={video1} type="video/mp4" />
            Your browser does not support the video tag.
          </Video>
          <Video autoPlay loop muted>
            <source src={video2} type="video/mp4" />
            Your browser does not support the video tag.
          </Video>
          <Video autoPlay loop muted>
            <source src={video3} type="video/mp4" />
            Your browser does not support the video tag.
          </Video>
        </VideoContainer>
      </Box>
      <Button
          variant="contained"
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate(-1)}
          sx={{ position: 'absolute', bottom: '20px', left: '20px'}}
        >
          Back
        </Button>
    </BlueContainer>
  );
};

export default Upload;
