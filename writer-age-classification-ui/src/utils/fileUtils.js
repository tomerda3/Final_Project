export const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        resolve(reader.result.split(',')[1]); // Remove the data URL part
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };
  