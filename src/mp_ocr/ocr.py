#!/usr/bin/env python3

import cv2
import logging
import numpy as np
import sys
import time

from itertools import chain, combinations

from . import utils, colors


def detect(image, knn, debug=False):
    # General steps are:
    #   1. Process the image
    #   2. Try and get the contours for the numbers
    #   3. Crop the image to those contours
    #   4. Perform contours one more time to get the numbers separately <- might change
    #   5. Detect the number individually
    logging.info(f"Detecting using image '{image.filename}'")
    processed_img = _preprocess_image(image)
    contour_groups = _extract_contours(processed_img, image)
    crops = _crop_contour_groups(image, contour_groups)

    # For debugging purposes
    if debug:
        cv2.imwrite(f"orig_{image.filename}", image.image)
        cv2.imwrite(f"bin_{image.filename}", processed_img)

        img = image.image
        for i, contours in enumerate(contour_groups):
            group_center = utils.get_contour_group_center(contours)
            font = cv2.FONT_HERSHEY_SIMPLEX

            img = cv2.fillPoly(img, pts=contours, color=colors.DIM_RED)
            img = cv2.drawContours(img, contours, -1, colors.RED, 2)

            cv2.putText(img, f"{i}", group_center, font, 1, colors.WHITE, 3, cv2.LINE_AA)
            cv2.putText(img, f"{i}", group_center, font, 1, colors.BLACK, 2, cv2.LINE_AA)

        cv2.imwrite(f"contours_{image.filename}", img)

        for i, cropped in enumerate(crops):
            cv2.imwrite(f"cropped_{i}_{image.filename}", cropped)


def _preprocess_image(image):
    logging.debug("Performing image preprocessing...")
    img = image.image

    # Apply gaussian blur
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to binary image
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return img


def _extract_contours(processed_img, image):
    # Find contours
    logging.debug(f"Getting contours...")
    _, contours, hier = cv2.findContours(
        processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    logging.debug(f"No. of contours: {len(contours)}")

    # Convert the contours into a single giant group
    contour_groups = [contours]

    # Start filtering out contours
    stages = {
        "Stage 1": ("Filtering bad contours", _stage_1_contour_groups_filter),
        "Stage 2": ("Comparing contours distance", _stage_2_contour_groups_filter),
        "Stage 3": (
            "Comparing contours line straightness",
            _stage_3_contour_groups_filter,
        ),
    }

    # Loop through the stages
    for stage_name, (stage_msg, stage_action) in stages.items():
        logging.debug(f"{stage_name}: {stage_msg}")

        before_count = sum(map(len, contour_groups))
        start = time.time()
        filtered_contours = stage_action(image, contour_groups)
        end = time.time()
        after_count = sum(map(len, filtered_contours))

        delta = before_count - after_count
        elapsed = (end - start) * 1000

        msg = f"{stage_name}: Removed {delta} contours, "
        msg += f"{after_count} remains. ({elapsed:.2f}ms)"
        logging.debug(msg)
        contour_groups = filtered_contours

    return contour_groups


def _crop_contour_groups(image, contour_groups):
    """
    Crop the image to the bounding rectangle of a group of contours
    """
    cropped_groups = list()
    img = image.image
    for contours in contour_groups:
        x1 = sys.maxsize
        x2 = -sys.maxsize
        y1 = sys.maxsize
        y2 = -sys.maxsize

        for contour in contours:
            cx1, cy1, cw, ch = cv2.boundingRect(contour)
            cx2 = cx1 + cw
            cy2 = cy1 + ch

            x1 = min(x1, cx1)
            x2 = max(x2, cx2)
            y1 = min(y1, cy1)
            y2 = max(y2, cy2)

        cropped = img[y1:y2, x1:x2]
        cropped_groups.append(cropped)
    return cropped_groups


def _stage_1_contour_groups_filter(image, contour_groups):
    height = image.height
    width = image.width
    area = image.area
    border_tolerance = 5

    filtered_contour_groups = list()

    for contours in contour_groups:
        filtered_contours = list()
        for i, contour in enumerate(contours):
            is_good, msg = _is_good_contour(contour, height, width, area)
            if is_good:
                logging.debug(f"Contour {i} is good")
                filtered_contours.append(contour)
            else:
                logging.debug(f"Contour {i} filtered: {msg}")

        if len(filtered_contours) > 0:
            filtered_contour_groups.append(filtered_contours)

    return filtered_contour_groups


def _is_good_contour(contour, height, width, area):
    is_good = False
    msg = None

    # Get contour features like area and aspect ratio
    _, _, w, h = cv2.boundingRect(contour)
    contour_area = cv2.contourArea(contour)
    contour_area_ratio = contour_area / area
    contour_aspect_ratio = h / w
    height_ratio = h / height
    width_ratio = w / width

    approx = utils.get_contour_approx(contour)
    num_approx = len(approx)
    _approx = np.squeeze(approx)

    # Edge case where there's only one point
    if _approx.ndim == 1:
        _approx = np.array([_approx])

    # Extract approximated x and y coordinates
    x_coords = _approx[:, 0]
    y_coords = _approx[:, 1]

    if False:
        pass

    ## Filter any contours where the height is less than the width
    elif h < w:
        msg = "height is less than width"

    # Filter any contours where the height is less than 1.2 times the
    # width
    elif contour_aspect_ratio < 1.2:
        msg =  f"contour aspect ratio is less than 1.2 "
        msg += f"({contour_aspect_ratio:.5f})"

    # Filter any contours where the height is greater than 3.4 times the
    # width
    elif contour_aspect_ratio > 3.4:
        msg = f"contour aspect ratio is more than 3.4 "
        msg += f"({contour_area_ratio:.5f})"

    # Filter any contours where the height is more than 75% the image
    # height
    elif height_ratio > 0.75:
        msg = f"contour height to image height ratio is more than 0.75 "
        msg = f"({height_ratio:.5f})"

    # Filter any contours where the width is more than 75% the image
    # width
    elif width_ratio > 0.75:
        msg = f"contour width to image width ratio is more than 0.75 "
        msg += f"({width_ratio:.5f})"

    # Filter any contours where the number of pixels is less than 70
    elif contour_area < 70:
        msg = f"contour area is less than 70 ({contour_area})"

    # Filter any contours where the area ratio is more than 10% of the
    # image
    elif contour_area_ratio > 0.1:
        msg = f"contour image area ratio is more than 0.1 "
        msg += f"({contour_area_ratio:.5f})"

    # Filter any contours where the area ratio is less than 0.012% of the
    # image
    elif contour_area_ratio < 0.0012:
        msg = f"contour image area ratio is less than 0.0012 "
        msg += f"({contour_area_ratio:.5f})"

    # Filter any contours where the number of approximated points are less
    # than 10
    elif num_approx < 10:
        msg = f"contour approximate datapoints count is less than 10 "
        msg += f"({num_approx})"

    # Filter any contours where the number of approximated points are more
    # than 1000
    elif num_approx > 1000:
        msg = f"contour approximate datapoints count is more than 1000 "
        msg += f"({num_approx})"

    # Filter any contours where they lie on the edges of the image
    elif (
        any(0 == i for i in x_coords)
        or any(0 == i for i in y_coords)
        or any(width == i for i in x_coords)
        or any(height == i for i in y_coords)
    ):
        msg = "contour is on the edge of the image"

    # Passed all checks
    else:
        is_good = True

    return is_good, msg


def _stage_2_contour_groups_filter(image, contour_groups):
    # Limit the distance between to the contours to 1/12th of the image
    # hypotenuse
    height = image.height
    width = image.width
    distance_limit = np.hypot(height, width) // 12
    logging.debug(f"Distance limit: {distance_limit:.2f}")

    filtered_contour_groups = list()
    for contours in contour_groups:
        groups = _stage_2_contour_group_filter(contours, distance_limit)
        for group in groups:
            filtered_contour_groups.append(group)
    return filtered_contour_groups


def _stage_2_contour_group_filter(contours, distance_limit):
    msg_fmt = "Distance between contour {} and {} is good"
    filtered_msg_fmt = (
        "Distance between contour {} and {} is not within the limit ({:.2f})"
    )

    # Check if there's more than 1 contour
    if len(contours) < 2:
        logging.debug(f"Not enough contours in group to compare. Skipping...")

        # Need to encapsulate the contours within a group
        return [contours]

    # For each possible pairing of contours
    distances = list()
    for i, j in combinations(range(len(contours)), 2):
        # Get the approximated points of the two contours
        approx_1 = utils.get_contour_approx(contours[i])
        approx_2 = utils.get_contour_approx(contours[j])

        # Get the smallest possible points between all points of the 2 contour
        # approximates
        logging.debug(f"Calculating smallest distance between contour {i} and {j}")
        contour_distances = [
            (i, j, np.linalg.norm(p2 - p1)) for p2 in approx_2 for p1 in approx_1
        ]
        _contour_distances = np.array(contour_distances)
        distances.append(contour_distances[np.argmin(_contour_distances[:, 2])])

    # For each calculated distances
    groups = list()
    for i, j, distance in distances:
        # Skip if distance is more than the limit
        if distance > distance_limit:
            msg = filtered_msg_fmt.format(i, j, distance)
            logging.debug(msg)
            continue

        msg = msg_fmt.format(i, j)
        logging.debug(msg)

        # Find the group that contains either contour indexes
        dest_group = next((g for g in groups if any(i in c or j in c for c in g)), None)

        if dest_group is None:
            # If no group is found, create a new group containing the 2
            # contours
            groups.append([(i, j)])
        else:
            dest_group.append((i, j))

    # Check if there are no contours that are on the same relative x
    # coordinates. This is an edge case, so we would need to treat each contour
    # as it's own group.
    if len(groups) == 0:
        for i, j, _ in distances:
            # We can link each index to itself because they will be trimmed out
            groups.append([(i, i)])
            groups.append([(j, j)])

    # Convert group indicies to contours
    groups = _map_group_indexes_to_contours(groups, contours)
    return groups


def _stage_3_contour_groups_filter(image, contour_groups):
    filtered_contour_groups = list()
    for contours in contour_groups:
        groups = _stage_3_contour_group_filter(contours)
        for group in groups:
            filtered_contour_groups.append(group)
    return filtered_contour_groups

def _stage_3_contour_group_filter(contours):
    msg_fmt = "Angle between contour {} and {} is good"
    filtered_msg_fmt = (
        "Angle between contour {} and {} is not in range ({:.2f} degrees)"
    )

    # Check if there's more than 1 contour
    if len(contours) < 2:
        logging.debug(f"Not enough contours to compare. Skipping...")

        # Need to encapsulate the contours within a group
        return [contours]

    # Get all the contour centers
    centers = np.array([utils.get_contour_center(i) for i in contours])

    # For each possible pairing of contours
    angles = list()
    for i, j in combinations(range(len(contours)), 2):
        p1 = centers[i]
        p2 = centers[j]

        # Get the angle of the gradient of the line connecting the two points
        angles.append((i, j, utils.angle_between_points(p1, p2)))

    # For each calculated angle
    groups = list()
    for i, j, angle in angles:
        # Skip if angle is more than the limit
        if not utils.is_around(angle, 0, 20) and not utils.is_around(angle, 180, 20):
            msg = filtered_msg_fmt.format(i, j, angle)
            logging.debug(msg)
            continue

        msg = msg_fmt.format(i, j, angle)
        logging.debug(msg)

        # Find the group that contains either contour indexes
        dest_group = next((g for g in groups if any(i in c or j in c for c in g)), None)

        if dest_group is None:
            # If no group is found, create a new group containing the 2
            # contours
            groups.append([(i, j)])
        else:
            dest_group.append((i, j))

    # Check if there are no contours that are on the same relative x
    # coordinates. This is an edge case, so we would need to treat each contour
    # as it's own group.
    if len(groups) == 0:
        for i, j, _ in angles:
            # We can link each index to itself because they will be trimmed out
            groups.append([(i, i)])
            groups.append([(j, j)])

    # Convert group indicies to contours
    groups = _map_group_indexes_to_contours(groups, contours)
    return groups


def _map_group_indexes_to_contours(groups, contours):
    """
    Data structure of groups is an array of an array containing a pair of
    ints. The ints are the indexes of the contours array.
    """
    group_indexes = list()
    mapped_groups = list()

    # Prepare indexes
    for group in groups:
        # Get all of the indexes as a 1d array
        indexes = utils.flatten(group)
        indexes = utils.unique(indexes)
        indexes.sort()
        group_indexes.append(indexes)

    # We need to combine any groups that contains the same indexes
    group_indexes = utils.link_groups(group_indexes)

    for indexes in group_indexes:
        # Map the indexes to the contours array
        mapped_groups.append([contours[i] for i in indexes])

    return mapped_groups


def _largest_contour_group_by_area(groups):
    """
    Returns the contour group with the largest combined area

    Data structure of groups is an array of an array containing a list of
    contours. They need to be mapped from ints in
    _map_group_indexes_to_contours.
    """
    areas = np.array([sum([cv2.contourArea(i) for i in g]) for g in groups])
    largest_group = groups[np.argmax(areas)]
    return largest_group
