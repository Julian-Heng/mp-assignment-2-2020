#!/usr/bin/env python3


def list_contains_same_elements(_list):
    # From: https://stackoverflow.com/a/3844832
    # Accessed: 12/10/2020
    iterator = iter(_list)
    try:
        first = next(iterator)
    except StopIteration:
        return true
    return all(first == rest for rest in _list)
