import cv2
import numpy as np
from typing import Tuple, List
from tqdm import tqdm


class PreProcess:
    def __init__(self, shape: Tuple):
        self.image_shape = shape

    # TODO: Replace resizing with segmentation
    def resize_images(self, images):
        print("\nResizing images...")
        # processed_images = [cv2.resize(im.image_data, (self.image_shape[0], self.image_shape[1])) for im in images]
        processed_images = [cv2.resize(im, (self.image_shape[0], self.image_shape[1])) for im in images]
        for row in tqdm(processed_images):
            for i in range(len(row)):
                row[i] = row[i] / 255
        return np.array(processed_images)

    def arrange_labels_indexing_from_0(self, labels: List) -> List:
        return [x-1 for x in labels if 0 not in labels]

    def grayscale_images(self, images):
        print("\nTurning images to grayscale...")
        grayscale_images = []
        for image in tqdm(images):
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            grayscale_images.append(image)
        return grayscale_images

    def reverse_binarize_images(self, images, threshold_method=cv2.THRESH_BINARY_INV):
        print("\nReverse binarizing images...")
        binarized_images = []

        for image in tqdm(images):
            thresh, binary_image = cv2.threshold(image, 127, 255, threshold_method)
            binarized_images.append(binary_image)

        return binarized_images

    def crop_text_from_reversed_binary_images(self, images, min_black_pixels=1):  # TODO: Fix, might not crop correctly
        print("\nCropping text from images...")
        cropped_images = []
        for image in tqdm(images):

            # Find top and bottom borders
            top = 0
            bottom = image.shape[0]
            for row in image:
                if sum(row) >= min_black_pixels:
                    top = max(top, row.tolist().index(0))
                    break
            for row in image[::-1]:
                if sum(row) >= min_black_pixels:
                    bottom = min(bottom, image.shape[0] - row.tolist().index(0) - 1)
                    break

            # Find left and right borders
            left = 0
            right = image.shape[1]
            for col in image.T:
                if sum(col) >= min_black_pixels:
                    left = max(left, col.tolist().index(0))
                    break
            for col in image.T[::-1]:
                if sum(col) >= min_black_pixels:
                    right = min(right, image.shape[1] - col.tolist().index(0) - 1)
                    break

            # Crop the image
            cropped_image = image[top:bottom, left:right]
            cropped_images.append(cropped_image)

        return cropped_images
