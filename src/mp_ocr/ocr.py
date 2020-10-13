#!/usr/bin/env python3

import cv2
import logging
import numpy as np


class OCR:

    @staticmethod
    def detect(image, knn):
        logging.debug(f"Detecting using image '{image.filename}'")
        extracts = OCR._preprocess_image(image)

    @staticmethod
    def _preprocess_image(image):
        logging.debug("Performing image preprocessing...")
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
        _, contours, hier = cv2.findContours(
            img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Start filtering out contours
        logging.debug(f"No. of contours: {len(contours)}")
        logging.debug("Filtering bad contours...")
        contours = [
            j
            for i, j in enumerate(contours)
            if OCR._is_good_contour(j, i, height, width, area)
        ]

        cv2.imwrite(f"bin_{image.filename}", img)
        img = image.image
        img = cv2.fillPoly(img, pts=contours, color=(0, 0, 127))
        img = cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
        cv2.imwrite(image.filename, img)
        cv2.imwrite(f"orig_{image.filename}", image.image)

    @staticmethod
    def _is_good_contour(contour, index, height, width, area):
        msg = f"Contour {index} is good"
        filter_msg = f"Contour {index} filtered: {{}}"
        is_good = False

        _, _, w, h = cv2.boundingRect(contour)
        contour_area = cv2.contourArea(contour)
        contour_ratio = h / w
        height_ratio = h / height
        width_ratio = w / width

        contour_area_ratio = contour_area / area

        approx = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
        _approx = np.squeeze(approx)

        # Edge case where there's only one point
        if _approx.ndim == 1:
            _approx = np.array([_approx])

        # Extract approximated x and y coordinates
        x_coords = _approx[:, 0]
        y_coords = _approx[:, 1]

        # Filter any contours where the height is less than the width
        if h < w:
            msg = filter_msg.format("height is less than width")

        # Filter any contours where the height is less than 1.36 times the
        # width
        elif contour_ratio < 1.36:
            msg = filter_msg.format("contour aspect ratio is less than 1.36")

        # Filter any contours where the height is greater than 3.4 times the
        # width
        elif contour_ratio > 3.4:
            msg = filter_msg.format("contour aspect ratio is more than 3.4")

        # Filter any contours where the height is more than 75% the image
        # height
        elif height_ratio > 0.75:
            msg = filter_msg.format("contour height to image height ratio is more than 0.75")

        # Filter any contours where the width is more than 75% the image
        # width
        elif width_ratio > 0.75:
            msg = filter_msg.format("contour width to image width ratio is more than 0.75")

        # Filter any contours where the number of pixels is less than 70
        elif contour_area < 70:
            msg = filter_msg.format("contour area is less than 70")

        # Filter any contours where the area ratio is more than 7.5% of the
        # image
        elif contour_area_ratio > 0.075:
            msg = filter_msg.format("contour image area ratio is more than 0.075")

        # Filter any contours where the area ratio is less than 0.0134% of the
        # image
        elif contour_area_ratio < 0.00134:
            msg = filter_msg.format("contour image area ratio is less than 0.00134")

        # Filter any contours where the number of approximated points are less
        # than 10
        elif len(approx) < 10:
            msg = filter_msg.format("contour approximate datapoints count is less than 10")

        # Filter any contours where the number of approximated points are more
        # than 1000
        elif len(approx) > 1000:
            msg = filter_msg.format("contour approximate datapoints count is more than 1000")

        # Filter any contours where they are on the left and top border of the
        # image
        elif any(0 in i for i in _approx):
            msg = filter_msg.format("contour is on the edge of the image")
        elif any(i > width - 5 for i in x_coords):
            msg = filter_msg.format("contour is on the edge of the image")
        elif any(i > height - 5 for i in y_coords):
            msg = filter_msg.format("contour is on the edge of the image")

        # Passed all checks
        else:
            is_good = True

        logging.debug(msg)
        return is_good
