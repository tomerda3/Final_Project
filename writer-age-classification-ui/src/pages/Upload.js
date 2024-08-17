import React, { useState } from 'react';
import { Container, Box, Typography, Button, CircularProgress, styled, Select, MenuItem, Divider, FormControl } from '@mui/material';
import { uploadHandwriting } from '../services/apiService';
import video1 from '../assets/video1.mp4';
import video2 from '../assets/video2.mp4';
import video3 from '../assets/video3.mp4';
import backGroundVideo from '../assets/net.mp4';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import { useNavigate } from 'react-router-dom';

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

const Upload = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const navigate = useNavigate();
  const [dataSet, setDataSet] = useState('HHD'); 
  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (file) {
      setLoading(true);
      try {
        const response = await uploadHandwriting(file, dataSet, 'ConvNeXtXLarge');
        setResult(response); 
      } catch (error) {
        console.error('Error uploading file:', error);
      } finally {
        setLoading(false);
      }
    }
  };
  const handleDataSetChange = (event) => {
    setDataSet(event.target.value);
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
        <form onSubmit={handleSubmit}>
          <input type="file" onChange={handleFileChange} />
          <Box mt={2}>
          <Typography variant="h6" paragraph>
          Choose Data Set
          </Typography>
            <FormControl fullWidth>
              <Select
                labelId="data-set-label"
                value={dataSet}
                onChange={handleDataSetChange}
                displayEmpty
              >
                <MenuItem value="HHD">HHD</MenuItem>
                <MenuItem value="KHATT">KHATT</MenuItem>
              </Select>
            </FormControl>
          </Box>
          <Box mt={2}>
            <UploadButton type="submit" variant="contained">
              {loading ? <CircularProgress size={24} color="inherit" /> : 'Start Predict'}
            </UploadButton>
          </Box>
        </form>
        <Divider sx={{ my: 1, borderBottomWidth: 2, borderColor: '#1e90ff' }} />
        {result && (
          <Box mt={4}>
            <Typography variant="h6" gutterBottom>
              Result from Backend:
            </Typography>
            <Typography variant="h5">{result}</Typography>
          </Box>
        )}
      </Box>
      <Box mt="auto">
        <Divider sx={{ my: 1, borderBottomWidth: 2, borderColor: '#1e90ff' }} />
        <VideoContainer>
        <Typography variant="body1">{result}</Typography>
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
