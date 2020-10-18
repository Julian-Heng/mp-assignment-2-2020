#!/usr/bin/env python3

""" Package main """

import sys

from . import mp_ocr


if __name__ == "__main__":
    mp_ocr.main(sys.argv[1:])
