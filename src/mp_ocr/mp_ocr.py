#!/usr/bin/env python3

import logging

from argparse import ArgumentParser
from pathlib import Path

from . import image, ocr, train, utils


def parse_args(args):
    parser = ArgumentParser()

    parser.add_argument("-t", "--train", action="store_true", default=False)
    parser.add_argument(
        "-to", "--train-output", action="store", type=Path, default=Path("out.npz")
    )
    parser.add_argument("-c", "--classifier", action="store", type=Path, metavar="path")
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("images", nargs="+", action="store", type=image.Image)

    return parser.parse_args(args)


def main(args):
    config = parse_args(args)

    # Set up logger
    log_format = "%(filename)s:%(funcName)s:%(lineno)s: %(message)s"
    level = logging.DEBUG if config.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format)

    logging.debug(f"Command line arguments: {args}")

    if config.train:
        # Program is run under training mode, images are used to train
        logging.debug(f"Training mode")
        images = config.images
        outfile = config.train_output
        train.knn.make_from_images(images, outfile)

    else:
        # Program is otherwise running in classifying mode, images are used for
        # ocr
        logging.debug(f"Classifying mode")
        if config.classifier.is_file():
            logging.debug(f"Using classifier file '{config.classifier}'")
            images = config.images
            classifier = config.classifier
            knn, res = train.knn.make_from_file(classifier)

            for image in images:
                ocr.detect(image, knn, res, debug=config.debug)
        else:
            logging.error(f"Invalid classifier file: '{config.classifier}'")
