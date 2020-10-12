#!/usr/bin/env python3

import cv2
import numpy as np

from .utils import list_contains_same_elements


class KNN:

    @staticmethod
    def make_from_images(images, outfile=None):
        """ Makes a knn classifier from image """
        # Validate images:
        #   1. They need to be the same size
        res = [i.resolution for i in images]
        if not list_contains_same_elements(res):
            raise TypeError("In order to train images, they need to be the same size")

        # Process the images
        train = np.array([KNN._preprocess_image(i) for i in images])
        labels = np.array([[int(i.parentname)] for i in images])

        if outfile is not None:
            np.savez(str(outfile), train=train, labels=labels)

        return KNN.make_from_training_data(train, labels)

    @staticmethod
    def make_from_file(infile):
        with np.load(str(infile)) as data:
            train = data["train"]
            labels = data["labels"]
            return KNN.make_from_training_data(train, labels)

    @staticmethod
    def make_from_training_data(train, labels):
        knn = cv2.ml.KNearest_create()
        knn.train(train, cv2.ml.ROW_SAMPLE, labels)
        return knn

    @staticmethod
    def _preprocess_image(image):
        src_img = image.image
        res = image.resolution
        area = image.area
        img = np.zeros(res)

        # Get the connected component
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        _, src_img = cv2.threshold(
            src_img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(src_img)

        # Get the second largest area component
        idx = np.argpartition(stats[:, cv2.CC_STAT_AREA], -2)[-2]

        # Mask of the label and convert it to a single dimensional array
        img[labels == idx] = 255
        img = img.reshape(-1, area).astype(np.float32)
        img = np.squeeze(img)

        return img
