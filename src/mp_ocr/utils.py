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


def angle_between_points(p1, p2):
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


def sort_contours_by_x_coord(contours):
    # From: https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
    # Acessed 17/10/2020

    if len(contours) == 1:
        return contours

    boxes = [cv2.boundingRect(i) for i in contours]
    sort, _ = zip(*sorted(zip(contours, boxes), key=lambda x: x[1][0]))
    return sort


def remove_inner_contours(contours):
    if len(contours) == 1:
        return contours

    filtered_contours = list()
    boxes = [cv2.boundingRect(i) for i in contours]

    for i in range(len(boxes)):
        check = True
        for j in range(len(boxes)):
            if i == j:
                continue
            elif is_rect_inside_rect(boxes[i], boxes[j]):
                check = False
                break

        if check:
            filtered_contours.append(contours[i])

    return filtered_contours


def is_rect_inside_rect(rect_1, rect_2):
    x1, y1, w1, h1 = rect_1
    x2, y2, w2, h2 = rect_2

    return x1 >= x2 and y1 >= y2 and (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2)
