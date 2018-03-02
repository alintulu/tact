# -*- coding: utf-8 -*-

"""
Module containing miscellaneous utility functions.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


class BinaryTree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.val = None


def s_to_n(cat, w=None):
    """
    Calculate the signal-to-noise ratio.

    Parameters
    ----------
    cat : 1D array, shape=N
        Array containing labels describing whether an entry is signal (1 or
        True) or background (0 or False).
    w : 1D array, shape=N, optional
        Weights. If none, then equal weights are used.

    Returns
    -------
    float
        Signal-to-noise ratio
    """

    if w is None:
        signal = len(cat[cat == 1])
        noise = len(cat[cat == 0])
    else:
        signal = w[cat == 1].sum()
        noise = w[cat == 0].sum()

    return signal / noise
