import React, { useState } from 'react';
import { Container, Box, Typography, Button, TextField } from '@mui/material';
import { styled, keyframes } from '@mui/system';
import { uploadHandwriting } from '../services/apiService';

const floatingAnimation = keyframes`
  0% { transform: translateY(0); }
  50% { transform: translateY(-10px); }
  100% { transform: translateY(0); }
`;

const BackgroundElement = styled(Box)(({ theme }) => ({
  position: 'absolute',
  width: '100px',
  height: '100px',
  backgroundColor: theme.palette.primary.main,
  borderRadius: '50%',
  animation: `${floatingAnimation} 4s ease-in-out infinite`,
  opacity: 0.5,
}));

const UploadContainer = styled(Container)({
  position: 'relative',
  minHeight: '100vh',
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  flexDirection: 'column',
  textAlign: 'center',
  zIndex: 1,
});

const BackgroundWrapper = styled(Box)({
  position: 'absolute',
  width: '100%',
  height: '100%',
  overflow: 'hidden',
  zIndex: 0,
});

const FloatingBackground = () => (
  <BackgroundWrapper>
    <BackgroundElement sx={{ top: '20%', left: '10%' }} />
    <BackgroundElement sx={{ top: '40%', right: '15%' }} />
    <BackgroundElement sx={{ bottom: '20%', left: '25%' }} />
    <BackgroundElement sx={{ bottom: '30%', right: '20%' }} />
  </BackgroundWrapper>
);

const Upload = () => {
  const [file, setFile] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (file) {
      try {
        const response = await uploadHandwriting(file);
        console.log(response.data); // Handle response data
      } catch (error) {
        console.error('Error uploading file:', error);
      }
    }
  };

  return (
    <UploadContainer>
      <FloatingBackground />
      <Typography variant="h3" gutterBottom>
        Upload Handwriting
      </Typography>
      <Box component="form" onSubmit={handleSubmit} sx={{ mt: 3 }}>
        <TextField
          variant="outlined"
          type="file"
          onChange={handleFileChange}
          fullWidth
          sx={{ mb: 2 }}
        />
        <Button variant="contained" color="primary" type="submit">
          Upload
        </Button>
      </Box>
    </UploadContainer>
  );
};

export default Upload;
