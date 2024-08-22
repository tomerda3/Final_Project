import cv2
import numpy as np
from typing import Tuple, List, Any, Union

from cv2 import Mat
from tqdm import tqdm


class PreProcess:
    def __init__(self, shape: Tuple):
        self.image_shape = shape

    def resize_images(self, images) -> List[np.array]:
        resized_images = [cv2.resize(im, (self.image_shape[0], self.image_shape[1])) for im in images]
        expand_images = [image.reshape(224, 224, 1) for image in resized_images]
        return expand_images

    def patch_images(self, images, labels, segment_shape):
        # print("Patching images...")
        patched_images = []
        patched_labels = []

        for i in range(len(images)):
            image = images[i]
            label = labels[i]
            height = len(image)
            width = len(image[0])
            y = 0

            y_step = int(segment_shape[1] * 0.50)  # Overlap of 50% of the Y per patch
            x_step = int(segment_shape[0] * 0.50)  # Overlap of 50% of the X per patch

            while y <= height - y_step:
                if segment_shape[1] - y < y_step:
                    break
                x = 0
                while x < width - x_step:
                    if segment_shape[0] - x < x_step:
                        break
                    segment = image[y: y + segment_shape[1], x: x + segment_shape[0]].copy()
                    x += x_step
                    if segment.shape[0] < segment_shape[0] or segment.shape[1] < segment_shape[1]:
                        continue
                    patched_images.append(segment)
                    patched_labels.append(label)
                y += y_step

        return np.array(patched_images), patched_labels

    def arrange_labels_indexing_from_0(self, labels: List) -> List[int]:
        return [x - 1 for x in labels if 0 not in labels]

    def grayscale_and_binarize_images(self, images) -> List[Union[Mat, np.ndarray]]:
        binarized_images = []
        for image in tqdm(images):
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)
            binarized_images.append(thresh)

        return binarized_images

    def crop_text_from_reversed_binary_images(self, images, min_white_pixels=100) -> List[Union[Mat, np.ndarray]]:
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
