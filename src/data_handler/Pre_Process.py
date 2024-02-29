import os
import cv2
import numpy as np

class KHATTDataPreprocessor:
    def __init__(self, image_size=(128, 128)):
        self.image_size = image_size

    def load_images_from_folder(self, folder_path):
        images = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.tif') or filename.endswith(".tiff"):
                img_path = os.path.join(folder_path, filename)
                print(img_path)
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    images.append(img)
        return images

    def preprocess_image(self, img):
        # Resize to a consistent size
        resized_img = cv2.resize(img, self.image_size)
        # Apply thresholding (Otsu's method)
        _, binary_img = cv2.threshold(resized_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Normalize pixel values to [0, 1]
        normalized_img = binary_img / 255.0
        return normalized_img

    def preprocess_folder(self, folder_path):
        images = self.load_images_from_folder(folder_path)
        preprocessed_images = [self.preprocess_image(img) for img in images]
        return preprocessed_images

if __name__ == "__main__":
    BASE_LOCAL_DIR = r"C:\Users\tomer\סמסטר א\פרויקט גמר\מאגרי נתונים"
    # Construct full paths
    # file_path = os.path.join(BASE_LOCAL_DIR, r"KHATT_v1.0")
    # train_folder = os.path.join(file_path, r"LineImages_v1.0\FixedTextLineImages\Train")
    train_folder =r"C:\\Users\tomer\\סמסטר א\\פרויקט גמר\\מאגרי נתונים\\KHATT_v1.0\\LineImages_v1.0\\FixedTextLineImages\\Train"
    # current folder os.getcwd()
    train_folder = os.getcwd()
    train_folder = r"C:\Users\tomer\OneDrive\Desktop\Final_Project\KHATT_v1.0\LineImages_v1.0\\FixedTextLineImages\\Train"
    preprocessor = KHATTDataPreprocessor()
    preprocessed_train_images = preprocessor.preprocess_folder(train_folder)
    print(f"Preprocessed {len(preprocessed_train_images)} images from {train_folder}")
