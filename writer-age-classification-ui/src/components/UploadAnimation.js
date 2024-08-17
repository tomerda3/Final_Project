import React from 'react';
import { Box } from '@mui/material';
import { styled } from '@mui/system';

const AnimatedBackground = styled(Box)({
  position: 'absolute',
  top: 0,
  left: 0,
  width: '100%',
  height: '100%',
  background: 'linear-gradient(45deg, #FF5722, #2196F3)',
  backgroundSize: '400% 400%',
  animation: 'gradient 15s ease infinite',
  zIndex: -1,
  overflow: 'hidden',
  
  '@keyframes gradient': {
    '0%': { backgroundPosition: '0% 0%' },
    '50%': { backgroundPosition: '100% 100%' },
    '100%': { backgroundPosition: '0% 0%' },
  },
});

const UploadAnimation = () => {
  return <AnimatedBackground />;
};

export default UploadAnimation;
