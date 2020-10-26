#!/usr/bin/env python3

""" utils module """

import itertools
import sys

import cv2
import numpy as np


def list_contains_same_elements(_list):
    """Returns true if a list contains the exact same elements

    From: https://stackoverflow.com/a/3844832
    Accessed: 12/10/2020

    Parameters
    ----------
    _list : list
        A list containing elements

    Returns
    -------
    bool
        Determines if the given list contains the same elements
    """
    iterator = iter(_list)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in _list)


def is_around(num, target, tolerance=0):
    """Returns true if a number is around a number

    Parameters
    ----------
    num : int
        The input number
    target : int
        The target number to compare against

    Returns
    -------
    bool
        Determines if the given number is around a number
    """
    low = target - tolerance
    high = target + tolerance
    return high > num and num > low


def angle_of_points(p1, p2):
    """Calculates the angle of gradient between 2 points

    Parameters
    ----------
    p1 : tuple
        Points represented as a tuple in the form (x, y)
    p2 : tuple
        Points represented as a tuple in the form (x, y)

    Returns
    -------
    float
        The angle of the gradient between 2 points
    """
    x1, y1 = p1
    x2, y2 = p2
    return abs(np.rad2deg(np.arctan2(y2 - y1, x2 - x1))) % 180


def flatten(_list):
    """Converts an nd array to a 1d array

    Parameters
    ----------
    _list : list
        List containing elements

    Returns
    -------
    list
        A 1d array containing elements from the input list
    """
    return list(itertools.chain(*_list))


def unique(_list):
    """Remove duplicate values from a list

    Parameters
    ----------
    _list : list
        List containings elements

    Returns
    -------
    list
        A list without any repeated values
    """
    return list(dict.fromkeys(_list))


def get_contour_center(contour):
    """Finds the center coordinates of a contour

    Parameters
    ----------
    contour : ndarray
        A contour represented as numpy array returned from the opencv
        findContour method

    Returns
    -------
    tuple
        The coordinates of a contour center in the form (x, y)
    """
    m = cv2.moments(contour)
    try:
        x = int(m["m10"] / m["m00"])
        y = int(m["m01"] / m["m00"])
        return (x, y)
    except ZeroDivisionError:
        return (0, 0)


def get_contour_group_bounding_rect(contours):
    """Gets the bounding rectangle of multiple contours

    Parameters
    ----------
    contours : list
        List containing contours

    Returns
    -------
    tuple
        The bounding rectangle in the form (x, y, w, h)
    """
    b = np.array([rect_to_coords(cv2.boundingRect(i)) for i in contours])
    x1 = np.min(b[:,0])
    y1 = np.min(b[:,1])
    x2 = np.max(b[:,2])
    y2 = np.max(b[:,3])
    return x1, y1, x2 - x1, y2 - y1


def get_contour_approx(contour):
    """Gets the approximated polygonal curves of a contour

    Parameters
    ----------
    contour : ndarray
        A contour represented as numpy array returned from the opencv
        findContour method

    Returns
    -------
    ndarray
        The approximated polygonal curves points of the contour
    """
    epsilon = 0.001 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx


def link_groups(groups):
    """Combine sublists with other sublists if they contain any common elements

    Parameters
    ----------
    groups : list
        List contaning list of elements

    Returns
    -------
    list
        List containing list of joined elements
    """
    result = groups[:1]

    for group in groups[1:]:
        # Get the next row in results containing a common element
        row = next((r for r in result if any(g in r for g in group)), None)

        if row is None:
            # If no such row exist, add to results
            result.append(group)
        else:
            # Else join with row
            row += group

    # There will be duplicated values, so we apply unique to remove them
    result = [unique(i) for i in result]
    return result


def map_group_indexes_to_contours(groups, contours):
    """Maps a list containing indexes to contours

    Data structure of groups is an array of an array containing a pair of
    ints. The ints are the indexes of the contours array.

    Parameters
    ----------
    groups : list
        List containing lists of ints that are indexes to the contours list
    contours : list
        List containing contours

    Returns
    -------
    list
        List containing lists of contours
    """
    # Flatten and remove duplicate indexes and link any groups containing the
    # same indexes
    group_indexes = link_groups([sorted(unique(flatten(i))) for i in groups])

    # Map the indexes to the contours
    mapped_groups = [[contours[j] for j in i] for i in group_indexes]
    return mapped_groups


def largest_contour_group_by_area(groups):
    """Returns the contour group with the largest combined area

    Data structure of groups is an array of an array containing a list of
    contours. They need to be mapped from ints in
    map_group_indexes_to_contours.

    Parameters
    ----------
    groups : list
        List containing lists of contours

    Returns
    -------
    index : int
        The index within the list of contours that has the largest combined
        area
    largest_group : list
        The contour group that represents the largest combined area
    """
    # Calculate the area of each group
    areas = np.array([sum([cv2.contourArea(i) for i in g]) for g in groups])

    # Get the largest area
    index = np.argmax(areas)
    largest_group = groups[index]
    return index, largest_group


def rect_to_coords(rect):
    """Returns the cartesian coordinates of a bounding rectangle

    Parameters
    ----------
    rect : tuple
        The bounding rectangle in a tuple in the form (x, y, w, h)

    Returns
    -------
    tuple
        The bounding rectangle in a tuple in the form (x1, y1, x2, y2)
    """
    if len(rect) < 4:
        raise TypeError("Not enough values in rect")
    x, y, w, h, *_ = rect
    return x, y, x + w, y + h
