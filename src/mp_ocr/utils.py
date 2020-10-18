#!/usr/bin/env python3

import cv2
import itertools
import numpy as np
import sys


def list_contains_same_elements(_list):
    # From: https://stackoverflow.com/a/3844832
    # Accessed: 12/10/2020
    iterator = iter(_list)
    try:
        first = next(iterator)
    except StopIteration:
        return true
    return all(first == rest for rest in _list)


def is_around(num, target, tolerance=0):
    return is_between(num, target - tolerance, target + tolerance)


def is_between(num, low, high):
    return num > low and high > num


def angle_of_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(np.rad2deg(np.arctan2(y2 - y1, x2 - x1))) % 180


def flatten(_list):
    return list(itertools.chain(*_list))


def unique(_list):
    return list(dict.fromkeys(_list))


def get_contour_center(contour):
    m = cv2.moments(contour)
    try:
        x = int(m["m10"] / m["m00"])
        y = int(m["m01"] / m["m00"])
        return (x, y)
    except ZeroDivisionError:
        return (0, 0)


def get_contour_group_bounding_rect(contours):
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

    return (x1, y1, x2 - x1, y2 - y1)


def get_contour_approx(contour):
    epsilon = 0.001 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx


def link_groups(groups):
    result = groups[:1]

    for group in groups[1:]:
        merged = False
        for row in result:
            if any(i in row for i in group):
                row += group
                merged = True
                break

        if not merged:
            result.append(group)

    result = [unique(i) for i in result]
    return result


def map_group_indexes_to_contours(groups, contours):
    """
    Data structure of groups is an array of an array containing a pair of
    ints. The ints are the indexes of the contours array.
    """
    group_indexes = list()
    mapped_groups = list()

    # Prepare indexes
    for group in groups:
        # Get all of the indexes as a 1d array
        indexes = flatten(group)
        indexes = unique(indexes)
        indexes.sort()
        group_indexes.append(indexes)

    # We need to combine any groups that contains the same indexes
    group_indexes = link_groups(group_indexes)

    for indexes in group_indexes:
        # Map the indexes to the contours array
        mapped_groups.append([contours[i] for i in indexes])

    return mapped_groups


def largest_contour_group_by_area(groups):
    """
    Returns the contour group with the largest combined area

    Data structure of groups is an array of an array containing a list of
    contours. They need to be mapped from ints in
    map_group_indexes_to_contours.
    """
    areas = np.array([sum([cv2.contourArea(i) for i in g]) for g in groups])
    index = np.argmax(areas)
    largest_group = groups[index]
    return index, largest_group
