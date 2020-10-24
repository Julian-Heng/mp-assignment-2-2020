#!/usr/bin/env python3

""" mp_ocr core application """

import logging
import sys

from argparse import ArgumentParser
from pathlib import Path

from . import image, ocr, train


def parse_args(args):
    """Parse command line arguments

    Parameters
    ----------
    args : list
        The command line arguments list

    Returns
    -------
    namespace
        A namespace object from argparse
    """
    parser = ArgumentParser()

    # Set up argument parser
    parser.add_argument("-t", "--train", action="store_true", default=False)
    parser.add_argument(
        "-to", "--train-output", action="store", type=Path,
        default=Path("out.npz")
    )
    parser.add_argument(
        "-c", "--classifier", action="store", type=Path, metavar="path"
    )
    parser.add_argument(
        "-o", "--output", action="store", type=Path, default=Path()
    )
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("images", nargs="+", action="store", type=image.Image)

    # Parse arguments
    return parser.parse_args(args)


def main(args):
    """Application main

    Parameters
    ----------
    args : list
        The command line arguments list
    """
    # Parse command line arguments
    config = parse_args(args)

    # Set up logger
    log_format = "%(filename)s:%(funcName)s:%(lineno)s: %(message)s"
    level = logging.DEBUG if config.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format)
    logging.debug("Command line arguments: %s", args)

    if config.train:
        # Program is run under training mode, images are used to train
        logging.debug("Training mode")
        images = config.images
        outfile = config.train_output
        train.knn.make_from_images(images, outfile)

    else:
        # Program is otherwise running in classifying mode, images are used for
        # ocr
        logging.debug("Classifying mode")

        # Check if arguments are valid
        if not config.classifier.is_file():
            logging.error("Invalid classifier file '%s'", config.classifier)
            sys.exit(1)

        if config.output.is_file():
            logging.error("Output path '%s' is not a directory", config.output)
            sys.exit(1)

        # Setup output directory
        if not config.output.exists():
            logging.info("Creating output path '%s'", config.output)
            config.output.mkdir(parents=True, exist_ok=True)

        logging.info("Using output path '%s'", config.output)

        # Load classifier file
        logging.debug("Using classifier file '%s'", config.classifier)
        images = config.images
        classifier = config.classifier
        knn, res = train.knn.make_from_file(classifier)

        # Detect digits
        for img in images:
            ocr.detect(
                img, knn, res, out=config.output, debug=config.debug
            )
