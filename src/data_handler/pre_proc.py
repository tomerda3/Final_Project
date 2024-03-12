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
