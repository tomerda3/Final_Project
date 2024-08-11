import React, { useState } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  CardMedia,
  Typography,
  Box,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  IconButton,
  Button,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import { useNavigate } from 'react-router-dom';
import { styled } from '@mui/system';

import vgg16Image from '../assets/vgg16.jpg'; // Adjust the paths as needed
import vgg19Image from '../assets/vgg19.jpg';
import efficientNetImage from '../assets/efficientnet.jpg';
import xceptionImage from '../assets/xception.jpg';
import resNetImage from '../assets/resnet.jpg';
import convNextXLImage from '../assets/convNextXLImage.jpg';
const FloatingCard = styled(Card)({
  transition: 'transform 0.3s, box-shadow 0.3s',
  '&:hover': {
    transform: 'translateY(-10px)',
    boxShadow: '0 8px 16px rgba(0, 0, 0, 0.2)',
  },
});

const modelData = [
  {
    name: 'Xception',
    description: 'Xception stands for "Extreme Inception" and is a deep convolutional neural network architecture inspired by the Inception architecture. It is known for its depth-wise separable convolutions, which aim to capture both spatial and channel-wise correlations in the data.',
    parameters: 'Tens of millions to over a hundred million',
    layers: 'Convolutional, depth-wise separable convolutional, and pooling layers, followed by fully connected layers for classification',
    image: xceptionImage,
  },
  {
    name: 'VGG16',
    description: 'VGG16 is a convolutional neural network architecture characterized by its simplicity, consisting of 16 weight layers, including 13 convolutional layers and 3 fully connected layers.',
    parameters: 'Approximately 138 million',
    layers: 'Convolutional layers followed by max-pooling layers, with fully connected layers at the end for classification',
    image: vgg16Image,
  },
  {
    name: 'VGG19',
    description: 'VGG19 is an extension of VGG16, featuring 19 weight layers, including 16 convolutional layers and 3 fully connected layers.',
    parameters: 'Around 144 million',
    layers: 'Convolutional layers followed by max-pooling layers and fully connected layers for classification',
    image: vgg19Image,
  },
  {
    name: 'ResNet',
    description: 'ResNet is a deep neural network architecture designed to address the problem of vanishing gradients in very deep networks. It introduces skip connections or shortcuts that allow gradients to flow more easily during training.',
    parameters: 'Tens of millions to over a hundred million',
    layers: 'Residual blocks, each containing several convolutional layers along with skip connections',
    image: resNetImage,
  },
  {
    name: 'EfficientNet',
    description: 'EfficientNet is a family of convolutional neural network architectures designed to achieve state-of-the-art accuracy with fewer parameters and FLOPs compared to other architectures. It uses a compound scaling method to scale up the network\'s depth, width, and resolution simultaneously.',
    parameters: 'Depends on the specific variant and scaling factor chosen',
    layers: 'Convolutional layers with various widths and depths determined by the scaling method',
    image: efficientNetImage,
  },
  {
    name: 'ConvNeXT XL',
    description: 'ConvNeXT XL is a large-scale convolutional neural network designed to be a general-purpose image classification model. It builds upon the design principles of ConvNeXT, focusing on optimizing performance and efficiency in image recognition tasks.',
    parameters: 'Approximately 1 billion',
    layers: 'Consists of a series of convolutional layers with varying kernel sizes, along with normalization and activation functions. It features an improved architecture compared to previous models, including enhanced depth and width.',
    image: convNextXLImage,
  }
  
];

const Models = () => {
  const [open, setOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);
  const navigate = useNavigate();

  const handleClickOpen = (model) => {
    setSelectedModel(model);
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
    setSelectedModel(null);
  };

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
        Models
        <Divider />
      </Typography>
      <Box my={2} />
      <Grid container spacing={4}>
        {modelData.map((model, index) => (
          <Grid item key={index} xs={12} sm={6} md={4}>
            <FloatingCard onClick={() => handleClickOpen(model)}>
              {model.image && (
                <CardMedia
                  component="img"
                  height="200"
                  image={model.image}
                  alt={model.name}
                />
              )}
              <CardContent>
                <Typography gutterBottom variant="h5" component="div">
                  {model.name}
                </Typography>
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
      {selectedModel && (
        <Dialog
          open={open}
          onClose={handleClose}
          fullWidth
          maxWidth="md"
          aria-labelledby="model-dialog-title"
        >
          <DialogTitle id="model-dialog-title">
            {selectedModel.name}
            <IconButton
              aria-label="close"
              onClick={handleClose}
              sx={{ position: 'absolute', right: 8, top: 8 }}
            >
              <CloseIcon />
            </IconButton>
          </DialogTitle>
          <DialogContent>
            <DialogContentText>
              {selectedModel.description}
            </DialogContentText>
            <Box mt={2}>
              <Typography variant="body2">
                <strong>Parameters:</strong> {selectedModel.parameters}
              </Typography>
              <Typography variant="body2">
                <strong>Layers:</strong> {selectedModel.layers}
              </Typography>
            </Box>
          </DialogContent>
        </Dialog>
      )}
    </Container>
  );
};

export default Models;
