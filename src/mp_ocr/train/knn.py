#!/usr/bin/env python3

import cv2
import logging
import numpy as np

from ..utils import list_contains_same_elements


def make_from_images(images, outfile=None):
    """ Makes a knn classifier from image """
    logging.debug("Making knn from images")

    # Validate images:
    #   1. They need to be the same size
    logging.debug("Checking for resolution")
    res = [i.resolution for i in images]
    if not list_contains_same_elements(res):
        raise TypeError("In order to train images, they need to be the same size")

    # Process the images
    train = np.array([_preprocess_image(i) for i in images])
    labels = np.array([[int(i.parentname)] for i in images])

    knn = make_from_training_data(train, labels)

    if knn and outfile is not None:
        logging.debug("Writing train and labels to file")
        np.savez(str(outfile), train=train, labels=labels)

    return knn


def make_from_file(infile):
    logging.debug(f"Loading knn train and labels from '{infile}'")
    with np.load(str(infile)) as data:
        train = data["train"]
        labels = data["labels"]
        return make_from_training_data(train, labels)


def make_from_training_data(train, labels):
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, labels)
    return knn


def _preprocess_image(image):
    logging.debug(f"Training using image '{image.filename}'")
    src_img = image.image
    res = image.resolution
    area = image.area
    img = np.zeros(res)

    # Get the connected component
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    _, src_img = cv2.threshold(src_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(src_img)

    # Get the second largest area component
    idx = np.argpartition(stats[:, cv2.CC_STAT_AREA], -2)[-2]

    # Mask of the label and convert it to a single dimensional array
    img[labels == idx] = 255
    img = img.reshape(-1, area).astype(np.float32)
    img = np.squeeze(img)

    return img
