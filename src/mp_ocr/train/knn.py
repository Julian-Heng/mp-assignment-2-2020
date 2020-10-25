#!/usr/bin/env python3

""" knn module """

import logging

from pathlib import Path

import cv2
import numpy as np

from ..utils import list_contains_same_elements


def make_from_images(images, outfile=None, debug=False):
    """Makes a knn classifier from image

    Parameters
    ----------
    images : list
        A list of images to train against
    outfile : Path, optional
        The destination file for storing the training and label data
    debug : bool, optional
        Toggles debug mode

    Returns
    -------
    knn : ml_KNearest
        The opencv knn model created using the training images

    Raises
    ------
    TypeError
        Raised when a not all the training images have the same resolution
    """
    logging.debug("Making knn from images")

    # Validate images:
    #   1. They need to be the same size
    logging.debug("Checking for resolution")
    res = [i.resolution for i in images]
    if not list_contains_same_elements(res):
        raise TypeError(
            "In order to train images, they need to be the same size"
        )

    # Process the images
    out = outfile.parent
    train = np.array([_preprocess_image(i, out, debug) for i in images])
    labels = np.array([[int(i.parentname)] for i in images])

    knn = make_from_training_data(train, labels)

    if knn and outfile is not None:
        logging.debug("Writing train and labels to file '%s'", outfile)
        np.savez(str(outfile), train=train, labels=labels, res=res[0])

    return knn


def make_from_file(infile):
    """Makes a knn classifier from file

    Parameters
    ----------
    infile : Path
        The clasifier file

    Returns
    -------
    knn : ml_KNearest
        The opencv knn model created using the input classifier file

    Raises
    ------
    TypeError
        Raised when infile does not exist
    """
    if not infile.exists():
        raise TypeError(f"File '{str(infile)}' does not exist")

    logging.debug("Loading knn train and labels from '%s'", infile)
    with np.load(str(infile)) as data:
        train = data["train"]
        labels = data["labels"]
        res = data["res"]
        return make_from_training_data(train, labels), res


def make_from_training_data(train, labels):
    """Makes a knn classifier using training data and labels

    Parameters
    ----------
    train : ndarray
        The training data used to train the knn classifier
    labels : ndarray
        The labels for each training data

    Returns
    -------
    knn : ml_KNearest
        The opencv knn model created using the training data and labels
    """
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, labels)
    return knn


def _preprocess_image(image, out=Path(), debug=False):
    """Prepare the image for training

    Parameters
    ----------
    img : ndarray
        The image as a ndarray to perform the preprocessing on
    out : Path, optional
        The output directory to write results to
    debug : bool, optional
        Toggles debug mode

    Returns
    -------
    img : ndarray
        The preprocessed image
    """
    logging.debug("Training using image '%s'", image.filename)
    src_img = image.image
    res = image.resolution
    area = image.area
    img = np.zeros(res)

    # Get the connected component
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    _, src_img = cv2.threshold(
        src_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    _, labels, stats, _ = cv2.connectedComponentsWithStats(src_img)

    # Get the second largest area component
    idx = np.argpartition(stats[:, cv2.CC_STAT_AREA], -2)[-2]

    # Mask of the label and convert it to a single dimensional array
    img[labels == idx] = 255

    # For debugging purposes
    if debug:
        fname = out.joinpath(f"DEBUG_training_orig_{image.filename}")
        logging.debug("Writing '%s'", fname)
        cv2.imwrite(str(fname), image.image)

        fname = out.joinpath(f"DEBUG_training_bin_{image.filename}")
        logging.debug("Writing '%s'", fname)
        cv2.imwrite(str(fname), src_img)

        fname = out.joinpath(f"DEBUG_training_{image.filename}")
        logging.debug("Writing '%s'", fname)
        cv2.imwrite(str(fname), img)

    img = img.reshape(-1, area).astype(np.float32)
    img = np.squeeze(img)

    return img
