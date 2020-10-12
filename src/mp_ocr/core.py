#!/usr/bin/env python3

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
    parser.add_argument("images", nargs="+", action="store", type=Image)

    return parser.parse_args(args)


def main(args):
    config = parse_args(args)
    if config.train:
        images = config.images
        outfile = config.train_output
        KNN.make_from_images(images, outfile)
    else:
        if config.classifier.is_file():
            images = config.images
            classifier = config.classifier
            knn = KNN.make_from_file(classifier)

            for image in images:
                OCR.detect(image, knn)
        else:
            print("Classifier provided is not a file")
