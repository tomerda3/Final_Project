import cv2
import numpy as np
from typing import Tuple, List, Any

from cv2 import Mat
from tqdm import tqdm


class PreProcess:
    def __init__(self, shape: Tuple):
        self.image_shape = shape

    def resize_images(self, images) -> List[np.array]:
        return [cv2.resize(im, (self.image_shape[0], self.image_shape[1])) for im in images]

    def patch_images(self, images, labels, segment_shape) -> np.array:
        patched_images = []
        patched_labels = []

        y_step = int(segment_shape[1] * 0.50)  # Overlap of 50% in Y
        x_step = int(segment_shape[0] * 0.50)  # Overlap of 50% in X

        for image, label in zip(images, labels):
            height, width = len(image), len(image[0])

            for y in range(0, height - segment_shape[1] + 1, y_step):
                for x in range(0, width - segment_shape[0] + 1, x_step):
                    segment = image[y: y + segment_shape[1], x: x + segment_shape[0]].copy()
                    patched_images.append(segment)
                    patched_labels.append(label)

        return np.array(patched_images), patched_labels

    def arrange_labels_indexing_from_0(self, labels: List) -> List[int]:
        return [x - 1 for x in labels if 0 not in labels]

    def grayscale_and_binarize_images(self, images) -> List[Mat | np.ndarray[Any, np.dtype] | np.ndarray]:
        binarized_images = []
        for image in tqdm(images):
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)
            binarized_images.append(thresh)

        return binarized_images

    def crop_text_from_reversed_binary_images(self, images, min_white_pixels=100) -> List[Mat | np.ndarray[Any, np.dtype] | np.ndarray]:
        WHITE = 255  # Assuming the images are in the range of 0 to 255

        cropped_images = []
        for image in tqdm(images):
            # Find top and bottom borders
            top = 0
            bottom = image.shape[0]
            for row in range(image.shape[0]):
                if np.count_nonzero(image[row] == WHITE) >= min_white_pixels:
                    top = max(top, row)
                    break
            for row in range(image.shape[0] - 1, -1, -1):
                if np.count_nonzero(image[row] == WHITE) >= min_white_pixels:
                    bottom = min(bottom, row)
                    break

            # Find left and right borders
            left = 0
            right = image.shape[1]
            for col in range(image.shape[1]):
                if np.count_nonzero(image[:, col] == WHITE) >= min_white_pixels:
                    left = max(left, col)
                    break
            for col in range(image.shape[1] - 1, -1, -1):
                if np.count_nonzero(image[:, col] == WHITE) >= min_white_pixels:
                    right = min(right, col)
                    break

            # Crop the image
            cropped_image = image[top:bottom, left:right]
            cropped_images.append(cropped_image)

        return cropped_images
