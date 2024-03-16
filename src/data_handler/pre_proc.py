import cv2
import numpy as np
from typing import Tuple, List


class PreProcess:
    def __init__(self, shape: Tuple):
        self.image_shape = shape

    # TODO: Replace resizing with segmentation
    def resize_images(self, images):
        processed_images = [cv2.resize(im.image_data, (self.image_shape[0], self.image_shape[1])) for im in images]
        for row in processed_images:
            for i in range(len(row)):
                row[i] = row[i] / 255
        return np.array(processed_images)

    def arrange_labels_indexing_from_0(self, labels: List) -> List:
        return [x-1 for x in labels if 0 not in labels]

    def crop_text_from_images(self, images, min_black_pixels=1):
        cropped_images = []
        for image in images:
            # Threshold to binary image (black = 0, white = 255)
            thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]

            # Find top and bottom borders
            top = 0
            bottom = thresh.shape[0]
            for row in thresh:
                if sum(row) >= min_black_pixels * 255:
                    top = max(top, row.tolist().index(0))
                    break
            for row in thresh[::-1]:
                if sum(row) >= min_black_pixels * 255:
                    bottom = min(bottom, thresh.shape[0] - row.tolist().index(0) - 1)
                    break

            # Find left and right borders
            left = 0
            right = thresh.shape[1]
            for col in thresh.T:
                if sum(col) >= min_black_pixels * 255:
                    left = max(left, col.tolist().index(0))
                    break
            for col in thresh.T[::-1]:
                if sum(col) >= min_black_pixels * 255:
                    right = min(right, thresh.shape[1] - col.tolist().index(0) - 1)
                    break

            # Crop the image
            cropped_image = image[top:bottom, left:right]
            cropped_images.append(cropped_image)

        return cropped_images