#!/usr/bin/env python3

import cv2
import numpy as np


class OCR:

    @staticmethod
    def detect(image, knn):
        extracts = OCR._preprocess_image(image)

    @staticmethod
    def _preprocess_image(image):
        img = image.image
        height = image.height
        width = image.width
        area = image.area

        # Apply gaussian blur
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Convert to binary image
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Find contours
        _, contours, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Start filtering out contours
        contours = [i for i in contours if OCR._is_good_contour(i, height, width, area)]

        cv2.imwrite(f"bin_{image.filename}", img)
        img = image.image
        img = cv2.fillPoly(img, pts=contours, color=(0, 0, 127))
        img = cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
        cv2.imwrite(image.filename, img)
        cv2.imwrite(f"orig_{image.filename}", image.image)

    @staticmethod
    def _is_good_contour(contour, height, width, area):
        _, _, w, h = cv2.boundingRect(contour)
        contour_area = cv2.contourArea(contour)
        contour_ratio = h / w
        height_ratio = h / height
        width_ratio = w / width

        contour_area_ratio = contour_area / area
        approx = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
        # Filter any contours where the height is less than the width
        if h < w:
            return False

        # Filter any contours where the height is less than 1.25 times the
        # width
        if contour_ratio < 1.36:
            return False

        # Filter any contours where the height is greater than 3.4 times the
        # width
        if contour_ratio > 3.4:
            return False

        # Filter any contours where the height is more than 75% the image
        # height
        if height_ratio > 0.75:
            return False

        # Filter any contours where the width is more than 75% the image
        # width
        if width_ratio > 0.75:
            return False

        # Filter any contours where the number of pixels is less than 70
        if contour_area < 70:
            return False

        # Filter any contours where the area ratio is more than 7.5% of the
        # image
        if contour_area_ratio > 0.075:
            return False

        # Filter any contours where the area ratio is less than 0.0134% of the
        # image
        if contour_area_ratio < 0.00134:
            return False

        # Filter any contours where the number of approximated points are less
        # than 10
        if len(approx) < 10:
            return False

        # Filter any contours where the number of approximated points are more
        # than 1000
        if len(approx) > 1000:
            return False

        _approx = np.squeeze(approx)
        x_coords = _approx[:, 0]
        y_coords = _approx[:, 1]

        # Filter any contours where they are on the left and top border of the
        # image
        if any(0 in i for i in _approx):
            return False
        if any(i > width - 5 for i in x_coords):
            return False
        if any(i > height - 5 for i in y_coords):
            return False

        return True
