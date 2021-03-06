#!/usr/bin/env python3

""" ocr module """

import itertools
import logging
import re
import time

from pathlib import Path

import cv2
import numpy as np

from . import utils, colors


def detect(image, knn, knn_res, out=Path(), debug=False):
    """Detects the digits of a given image

    General steps are:
        1. Process the image
        2. Try and get the contours for the numbers
        3. Crop the image to those contours
        4. Perform connected components on the crop
        5. Detect the number individually

    Parameters
    ----------
    image : Image
        The image object to detect from
    knn : ml_KNearest
        The opencv knn model used to predict the digits
    knn_res : ndarray
        The trained image's resolution in a numpy array in the form [w, h]
    out : Path, optional
        The output directory to write results to
    debug : bool, optional
        Toggles debug mode
    """
    logging.info("Detecting using image '%s'", image.filename)

    # Start timer
    start = time.time()

    # Image preprocessing and extraction
    processed_img = _preprocess_image(image.image)
    contour_groups = _extract_contours(processed_img, image)
    crops = _crop_contour_groups(image, contour_groups)

    # Check if we've extracted anything
    if len(contour_groups) == 0:
        logging.info("Unable to detect any contours!!")
        return

    # Get largest contour
    i, contours = utils.largest_contour_group_by_area(contour_groups)
    crop = crops[i].copy()

    # Resolution spec needs to be reversed and be in a tuple for cv2.resize
    knn_res = tuple(knn_res[::-1])

    # Detect digits and write results
    digits = _detect_digits(image, crop, knn, knn_res, out, debug)
    logging.debug("Detected digits: %s", digits)

    end = time.time()
    elapsed = (end - start) * 1000
    logging.info("Finished in %.2fms", elapsed)

    _write_results(image, contours, crop, digits, out)

    # For debugging purposes
    if debug:
        _write_results_debug(image, processed_img, contour_groups, crops, out)


def _preprocess_image(img):
    """Preprocess an image for digits detection

    This function will apply a Gaussian filter, followed by an Otsu
    binary threshold filter.

    Parameters
    ----------
    img : ndarray
        The image as a ndarray to perform the preprocessing on

    Returns
    -------
    img : ndarray
        The preprocessed image
    """
    logging.debug("Performing image preprocessing...")

    # Apply gaussian blur
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to binary image
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return img


def _extract_contours(processed_img, image):
    """Extracts the most relevant contours for digit detection

    This function will try to find contours that fits a specific criterion that
    best matches digits.

    The filtering takes 3 stages:
        1. Filter on the features, such as width and height of the contour
        2. Group contours by distance
        3. Group contours by angle

    Parameters
    ----------
    processed_img : ndarray
        The preprocessed binary image
    image : Image
        The image object of the given image

    Returns
    -------
    contour_groups : list
        List containing lists of contours

    Notes
    -----
    The structure of the returned value is a list of groups of contours
    """
    # Find contours
    logging.debug("Getting contours...")
    ret = cv2.findContours(
        processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # findContours return values are different depending on the version of
    # OpenCV
    contours = ret[1 if len(ret) == 3 else 0]

    logging.debug("No. of contours: %d", len(contours))

    # Convert the contours into a single giant group
    contour_groups = [contours]

    # Start filtering out contours
    stages = {
        "Stage 1": ("Filtering bad contours", _stage_1_contour_groups_filter),
        "Stage 2": (
            "Comparing contours distance",
            _stage_2_contour_groups_filter
        ),
        "Stage 3": (
            "Comparing contours line straightness",
            _stage_3_contour_groups_filter,
        ),
    }

    # Loop through the stages
    logging.info("Started filtering...")
    total_start = time.time()
    for stage_name, (stage_msg, stage_action) in stages.items():
        logging.info("%s: %s", stage_name, stage_msg)

        start = time.time()
        filtered_contours = stage_action(image, contour_groups)
        end = time.time()

        # Calculate difference in contours
        before_count = sum(map(len, contour_groups))
        after_count = sum(map(len, filtered_contours))
        delta = before_count - after_count

        # Calculate time taken
        elapsed = (end - start) * 1000

        # Print statistics
        logging.debug(
            "%s: Removed %d contours, %d remains. (%.2fms)",
            stage_name, delta, after_count, elapsed
        )
        contour_groups = filtered_contours

    # Print statistics
    total_end = time.time()
    elapsed = (total_end - total_start) * 1000
    logging.debug("Finished filtering (%.2fms)", elapsed)

    return contour_groups


def _crop_contour_groups(image, contour_groups):
    """Crop the image to the bounding rectangle of a group of contours

    Parameters
    ----------
    image : Image
        The image object of the given image
    contour_groups : list
        List containing lists of contours

    Returns
    -------
    cropped_groups : list
        List containing the cropped contour groups
    """
    cropped_groups = list()
    img = image.image

    # For each contour group, get bounding rectangle of the group and crop
    for contours in contour_groups:
        x1, y1, x2, y2 = utils.rect_to_coords(
            utils.get_contour_group_bounding_rect(contours)
        )
        cropped = img[y1:y2, x1:x2]
        cropped_groups.append(cropped)

    return cropped_groups


def _detect_digits(image, crop, knn, knn_res, out=Path(), debug=False):
    """Detect the digits of an cropped image

    Parameters
    ----------
    image : Image
        The image object of the given image
    crop : ndarray
        The crop of the image containing the digit to detect
    knn : ml_KNearest
        The opencv knn model used to predict the digits
    knn_res : ndarray
        The trained image's resolution in a numpy array in the form [w, h]
    out : Path
        The output directory to write results to
    debug : bool, optional
        Toggles debug mode

    Returns
    -------
    str
        The detected digits
    """
    # Extract the connected components from the cropped image
    img, components, components_coords = _detect_digits_extract_components(
        crop
    )

    if debug:
        dest_fname = str(out.joinpath(f"DEBUG_cropped_bin_{image.filename}"))
        cv2.imwrite(dest_fname, img)

    # Begin detecting the different components
    digits = list()
    for i, (mask, coords) in enumerate(zip(components, components_coords)):
        x1, y1, x2, y2 = utils.rect_to_coords(coords)
        cv2.rectangle(crop, (x1, y1), (x2, y2), colors.RED, 2)

        # Prepare the mask for digit detection
        mask = _prepare_digit_mask(mask, coords, knn_res)

        if debug:
            dest_fname = f"DEBUG_component_{i}_{image.filename}"
            dest_fname = str(out.joinpath(dest_fname))
            cv2.imwrite(dest_fname, mask)

        mask = mask.reshape(-1, np.prod(knn_res)).astype(np.float32)

        # Predict
        ret, _, _, _ = knn.findNearest(mask, k=1)
        digits.append(str(int(ret)))

    return "".join(digits)


def _detect_digits_extract_components(crop):
    """Extract the connected components from a cropped image

    Parameters
    ----------
    crop : ndarray
        The crop of the image containing the digit to detect

    Returns
    -------
    img : ndarray
        The processed binary of the cropped image
    list
        A list containing the connected components
    list
        A list containing coordinates of the connected component
    """
    # Preprocess the cropped image and get the connected components
    img = _preprocess_image(crop)
    count, labels = cv2.connectedComponents(img)

    # Extract the individual components
    components = list()
    components_coords = list()
    for label in range(1, count):
        mask = np.array(labels, dtype=np.uint8)
        mask[labels == label] = 255
        mask[labels != label] = 0
        coords = cv2.boundingRect(mask)

        # Crop the component
        x1, y1, x2, y2 = utils.rect_to_coords(coords)
        mask = mask[y1:y2, x1:x2]

        components.append(mask)
        components_coords.append(coords)

    # Sort components by x coordinate
    components, components_coords = zip(
        *sorted(zip(components, components_coords), key=lambda x: x[1][0])
    )

    return img, components, components_coords


def _prepare_digit_mask(mask, coords, knn_res):
    """Process the connected components to resemble the training images

    Parameters
    ----------
    mask : ndarray
        The connected component of a digit
    coords : ndarray
        The bounding box coordinates of the mask from the original crop
    knn_res : ndarray
        The trained image's resolution in a numpy array in the form [w, h]

    Returns
    -------
    mask : ndarray
        The processed connected component
    """
    knn_w, knn_h = knn_res
    border_pad = 1

    _, _, w, h = coords

    # Detect any pixels on the border of the image
    borders = (mask[:, 0], mask[:, -1], mask[0, :], mask[-1, :])
    border_checks = [border_pad if 255 in line else 0 for line in borders]

    # Pad any borders if any pixel touches the edges
    mask = cv2.copyMakeBorder(
        mask, *border_checks, cv2.BORDER_CONSTANT, colors.BLACK
    )

    # Pad both width and height to retain aspect ratio after resizing
    pad_w = (int(knn_w * (h / knn_h)) - w) // 2
    pad_h = (int(knn_h * (w / knn_w)) - h) // 2

    pad_w = pad_w if pad_w > 0 else 0
    pad_h = pad_h if pad_h > 0 else 0

    mask = cv2.copyMakeBorder(
        mask, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, colors.BLACK
    )

    # Resize to the training image resolution
    mask = cv2.resize(mask, knn_res, interpolation=cv2.INTER_NEAREST)
    return mask


def _write_results(image, contours, crop, digits, out):
    """Write results to files

    Parameters
    ----------
    image : Image
        The image object of the given image
    contours : list
        List containing contours
    crop : ndarray
        The crop of the image containing the digit to detect
    digits : str
        The detected digits
    out : Path
        The output directory to write results to
    """
    # Extract image number from filename
    try:
        fname = re.findall(r"\d+", image.filename_without_extension)[0]
    except IndexError:
        fname = str(0)
    ext = image.extension

    box = utils.get_contour_group_bounding_rect(contours)

    # Write the crop
    dest_fname = str(out.joinpath(f"DetectedArea{fname}{ext}"))
    logging.info("Writing '%s'", dest_fname)
    cv2.imwrite(dest_fname, crop)

    # Write the bounding box specification
    dest_fname = out.joinpath(f"BoundingBox{fname}.txt")
    logging.info("Writing '%s'", dest_fname)
    with open(dest_fname, "w") as f:
        f.write(str(box))

    # Write the detected digits
    dest_fname = out.joinpath(f"House{fname}.txt")
    logging.info("Writing '%s'", dest_fname)
    with open(dest_fname, "w") as f:
        f.write(f"Building {digits}")


def _write_results_debug(image, processed_img, contour_groups, crops, out):
    """Write debug information to files

    Parameters
    ----------
    image : Image
        The image object of the given image
    processed_img : ndarray
        The preprocessed binary image
    contour_groups : list
        List containing lists of contours
    crops : list
        The list of crops of the image derived from the contour groups
    out : Path
        The output directory to write results to
    """
    img = image.image

    # Write original and binary image
    dest_fname = str(out.joinpath(f"DEBUG_orig_{image.filename}"))
    cv2.imwrite(dest_fname, img)

    dest_fname = str(out.joinpath(f"DEBUG_bin_{image.filename}"))
    cv2.imwrite(dest_fname, processed_img)

    # Draw each contour and label them accordingly
    for i, contours in enumerate(contour_groups):
        x, y, w, h = utils.get_contour_group_bounding_rect(contours)
        x += w // 2
        y += h // 2

        font = cv2.FONT_HERSHEY_SIMPLEX

        img = cv2.fillPoly(img, pts=contours, color=colors.DIM_RED)
        img = cv2.drawContours(img, contours, -1, colors.RED, 2)

        cv2.putText(img, f"{i}", (x, y), font, 1, colors.WHITE, 3, cv2.LINE_AA)
        cv2.putText(img, f"{i}", (x, y), font, 1, colors.BLACK, 2, cv2.LINE_AA)

    # Write contours extracted
    dest_fname = str(out.joinpath(f"DEBUG_contours_{image.filename}"))
    cv2.imwrite(dest_fname, img)

    # Write each crop
    for i, cropped in enumerate(crops):
        dest_fname = str(out.joinpath(f"DEBUG_cropped_{i}_{image.filename}"))
        cv2.imwrite(dest_fname, cropped)


def _stage_1_contour_groups_filter(image, contour_groups):
    """Filter contour groups by contour features

    Parameters
    ----------
    image : Image
        The image object of the given image
    contour_groups : list
        List containing lists of contours

    Returns
    -------
    filtered_contour_groups : list
        List containing lists of the remaining, non-filtered contours
    """
    height = image.height
    width = image.width
    area = image.area

    filtered_contour_groups = list()

    # For each contour groups
    for contours in contour_groups:
        filtered_contours = list()
        for i, contour in enumerate(contours):
            # Determine if the contour is good
            is_good, msg = _is_good_contour(contour, height, width, area)
            if is_good:
                logging.debug("Contour %d is good", i)
                filtered_contours.append(contour)
            else:
                logging.debug("Contour %d filtered: %s", i, msg)

        if len(filtered_contours) > 0:
            filtered_contour_groups.append(filtered_contours)

    return filtered_contour_groups


def _is_good_contour(contour, height, width, area):
    """Filter contour groups by contour features

    Parameters
    ----------
    contour : ndarray
        A contour represented as numpy array returned from the opencv
        findContour method
    height : int
        The height of the image
    width : int
        The width of the image
    area : int
        The area of the image

    Returns
    -------
    is_good : bool
        Determines if the contour should or should not be filtered
    msg : str
        The reason, if any, for the contour being filtered
    """
    is_good = False
    msg = None

    # Get contour features like area and aspect ratio
    _, _, w, h = cv2.boundingRect(contour)
    contour_area = cv2.contourArea(contour)
    contour_area_ratio = contour_area / area
    contour_aspect_ratio = h / w
    height_ratio = h / height
    width_ratio = w / width

    # Get contour approximates
    approx = np.squeeze(utils.get_contour_approx(contour))

    # Edge case where there's only one point
    if approx.ndim == 1:
        approx = np.array([approx])

    # Extract approximated x and y coordinates
    x_coords = approx[:, 0]
    y_coords = approx[:, 1]

    # Dummy condition
    if False:
        pass

    # Filter any contours where the height is less than the width
    elif h < w:
        msg = "height is less than width "
        msg += f"(h: {h}, w: {w})"

    # Filter any contours where the height is less than 1.2 times the
    # width
    elif contour_aspect_ratio < 1.2:
        msg = "contour aspect ratio is less than 1.2 "
        msg += f"({contour_aspect_ratio:.5f})"

    # Filter any contours where the height is greater than 3.4 times the
    # width
    elif contour_aspect_ratio > 3.4:
        msg = "contour aspect ratio is more than 3.4 "
        msg += f"({contour_area_ratio:.5f})"

    # Filter any contours where the height is more than 75% the image
    # height
    elif height_ratio > 0.75:
        msg = "contour height to image height ratio is more than 0.75 "
        msg = f"({height_ratio:.5f})"

    # Filter any contours where the width is more than 75% the image
    # width
    elif width_ratio > 0.75:
        msg = "contour width to image width ratio is more than 0.75 "
        msg += f"({width_ratio:.5f})"

    # Filter any contours where the number of pixels is less than 70
    elif contour_area < 70:
        msg = f"contour area is less than 70 ({contour_area})"

    # Filter any contours where the area ratio is more than 10% of the
    # image
    elif contour_area_ratio > 0.1:
        msg = "contour image area ratio is more than 0.1 "
        msg += f"({contour_area_ratio:.5f})"

    # Filter any contours where the area ratio is less than 0.12% of the
    # image
    elif contour_area_ratio < 0.0012:
        msg = "contour image area ratio is less than 0.0012 "
        msg += f"({contour_area_ratio:.5f})"

    # Filter any contours where they lie on the edges of the image
    elif (
        any(utils.is_around(i, 0, tolerance=5) for i in x_coords)
        or any(utils.is_around(i, 0, tolerance=5) for i in y_coords)
        or any(utils.is_around(i, width, tolerance=5) for i in x_coords)
        or any(utils.is_around(i, height, tolerance=5) for i in y_coords)
    ):
        msg = "contour is on the edge of the image"

    # Passed all checks
    else:
        is_good = True

    return is_good, msg


def _stage_2_contour_groups_filter(image, contour_groups):
    """Group contours by distances between contours

    Parameters
    ----------
    image : Image
        The image object of the given image
    contour_groups : list
        List containing lists of contours

    Returns
    -------
    filtered_contour_groups : list
        List containing lists of the remaining, non-filtered contours
    """
    # Limit the distance between to the contours to 1/12th of the image
    # hypotenuse
    height = image.height
    width = image.width
    dist_limit = np.hypot(height, width) // 12
    logging.debug("Distance limit: %.2f", dist_limit)

    return list(itertools.chain(
        *[_stage_2_contour_group_filter(i, dist_limit) for i in contour_groups]
    ))


def _stage_2_contour_group_filter(contours, distance_limit):
    """Helper function to split a given group of contours into multiple groups
    on the distance between contours

    Parameters
    ----------
    contours : list
        List containing contours
    distance_limit : int
        The distance limit to split contours by

    Returns
    -------
    groups : list
        List containing groups of contours
    """
    info_fmt = "Calculating smallest distance between contour %d and %d"
    msg_fmt = "Distance between contour %d and %d is good (%.2fpx)"
    filtered_msg_fmt = (
        "Distance between contour %d and %d is not within the limit (%.2fpx)"
    )

    # Check if there's only 1 contour
    if len(contours) < 2:
        logging.debug("Not enough contours in group to compare. Skipping...")

        # Need to encapsulate the contours within a group
        return [contours]

    # For each possible pairing of contours
    distances = list()
    for i, j in itertools.combinations(range(len(contours)), 2):
        # Get the approximated points of the two contours
        approx_1 = utils.get_contour_approx(contours[i])
        approx_2 = utils.get_contour_approx(contours[j])

        # Get the smallest possible distances between all points of the 2
        # contour approximates
        logging.debug(info_fmt, i, j)
        contour_distances = [
            (i, j, np.linalg.norm(p2 - p1))
            for p2 in approx_2 for p1 in approx_1
        ]
        _contour_distances = np.array(contour_distances)
        distances.append(
            contour_distances[np.argmin(_contour_distances[:, 2])]
        )

    # For each calculated distances
    groups = list()
    for i, j, distance in distances:
        # Skip if distance is more than the limit
        if distance > distance_limit:
            logging.debug(filtered_msg_fmt, i, j, distance)
            continue

        logging.debug(msg_fmt, i, j, distance)

        # Find the group that contains either contour indexes
        dest_group = (g for g in groups if any(i in c or j in c for c in g))
        dest_group = next(dest_group, None)

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
        # We can link each index to itself because they will be trimmed out
        groups = [[(i, i), (j, j)] for i, j, _ in angles]

    # Convert group indicies to contours
    groups = utils.map_group_indexes_to_contours(groups, contours)
    return groups


def _stage_3_contour_groups_filter(image, contour_groups):
    """Group contours by angles between contours

    Parameters
    ----------
    image : Image
        The image object of the given image
    contour_groups : list
        List containing lists of contours

    Returns
    -------
    filtered_contour_groups : list
        List containing lists of the remaining, non-filtered contours
    """
    return list(itertools.chain(
        *[_stage_3_contour_group_filter(i) for i in contour_groups]
    ))


def _stage_3_contour_group_filter(contours):
    """Helper function to split a given group of contours into multiple groups
    on the angle between contours

    Parameters
    ----------
    contours : list
        List containing contours

    Returns
    -------
    groups : list
        List containing groups of contours
    """
    info_fmt = "Calculating angle between contour %d and %d"
    msg_fmt = "Angle between contour %d and %d is good (%.2f degrees)"
    filtered_msg_fmt = (
        "Angle between contour %d and %d is not in range (%.2f degrees)"
    )

    # Check if there's only 1 contour
    if len(contours) < 2:
        logging.debug("Not enough contours to compare. Skipping...")

        # Need to encapsulate the contours within a group
        return [contours]

    # Get all the contour centers
    centers = np.array([utils.get_contour_center(i) for i in contours])

    # For each possible pairing of contours
    angles = list()
    for i, j in itertools.combinations(range(len(contours)), 2):
        p1 = centers[i]
        p2 = centers[j]

        logging.debug(info_fmt, i, j)

        # Get the angle of the gradient of the line connecting the two points
        angles.append((i, j, utils.angle_of_points(p1, p2)))

    # For each calculated angle
    groups = list()
    for i, j, angle in angles:
        # Skip if angle is not within range
        if (
            not utils.is_around(angle, 0, 20)
            and not utils.is_around(angle, 180, 20)
        ):
            logging.debug(filtered_msg_fmt, i, j, angle)
            continue

        logging.debug(msg_fmt, i, j, angle)

        # Find the group that contains either contour indexes
        dest_group = (g for g in groups if any(i in c or j in c for c in g))
        dest_group = next(dest_group, None)

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
        # We can link each index to itself because they will be trimmed out
        groups = [[(i, i), (j, j)] for i, j, _ in angles]

    # Convert group indicies to contours
    groups = utils.map_group_indexes_to_contours(groups, contours)
    return groups
