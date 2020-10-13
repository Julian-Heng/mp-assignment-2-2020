#!/usr/bin/env python3

import logging

from pathlib import Path
from argparse import ArgumentParser

from . import utils
from .image import Image
from .train import KNN
from .ocr import OCR


def parse_args(args):
    parser = ArgumentParser()

    parser.add_argument("-t", "--train", action="store_true", default=False)
    parser.add_argument(
        "-to", "--train-output", action="store", type=Path, default=Path("out.npz")
    )
    parser.add_argument("-c", "--classifier", action="store", type=Path, metavar="path")
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("images", nargs="+", action="store", type=Image)

    return parser.parse_args(args)


def main(args):
    config = parse_args(args)

    # Set up logger
    log_format = "%(filename)s:%(funcName)s:%(lineno)s: %(message)s"
    level = logging.DEBUG if config.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format)

    logging.debug(f"Command line arguments: {args}")

    # Program is run under training mode, images are used to train
    if config.train:
        logging.debug(f"Training mode")
        images = config.images
        outfile = config.train_output
        KNN.make_from_images(images, outfile)

    # Program is otherwise running in classifying mode, images are used for ocr
    else:
        logging.debug(f"Classifying mode")
        if config.classifier.is_file():
            logging.debug(f"Using classifier file '{config.classifier}'")
            images = config.images
            classifier = config.classifier
            knn = KNN.make_from_file(classifier)

            for image in images:
                OCR.detect(image, knn, debug=config.debug)
        else:
            logging.error(f"Invalid classifier file: '{config.classifier}'")
