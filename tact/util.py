# -*- coding: utf-8 -*-

"""
Module containing miscellaneous utility functions.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import collections
import itertools

import numpy as np


class BinaryTree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.val = None


def deep_update(d1, d2):
    """
    Adds key-value pairs in d2 to d1. Conflicts are resolved in favour of d2.

    Recurses into all values in d2 which belong to the collections.Mapping
    abstract base class.

    Parameters
    ----------
    d1 : collections.Mapping
        Base dictionary
    d2 : collections.Mapping
        Dictionary with updated values

    Returns
    -------
    d1 : collections.Mapping
        Updated dictionary
    """

    for k, v in d2.iteritems():
        if isinstance(v, collections.Mapping):
            d1[k] = deep_update(d1.get(k, {}), v)
        else:
            d1[k] = v
    return d1


def nodes(tree):
    """
    Return a list of values at every node of a tree.

    Parameters
    ----------
    tree : BinaryTree
        BinaryTree to extract nodes from.

    Returns
    -------
    nodelist : list
        List of values at tree nodes.
    """

    nodelist = []

    def _get_nodes(tree):
        """
        Build up a list of nodes.

        Parameters
        ----------
        tree : BinaryTree
            BinaryTree to extract nodes from.

        Returns
        -------
        None
        """

        nodelist.append(tree.val)
        try:
            _get_nodes(tree.left)
        except AttributeError:
            nodelist.append(tree.left)
        try:
            _get_nodes(tree.right)
        except AttributeError:
            nodelist.append(tree.right)

    _get_nodes(tree)

    return nodelist


def maenumerate(marr):
    """
    Multidimensional index iterator for masked arrays.

    Return an iterator yielding pairs of array coordinates and values, with
    masked values skipped.

    Parameters
    ----------
    marr : MaskedArray
      Input array.
    """

    for i, m in itertools.izip(np.ndenumerate(marr), ~marr.mask.ravel()):
        if m:
            yield i


def corrcoef(x, y=None, rowvar=True, fweights=None, aweights=None):
    """
    Return Pearson product-moment correlation coefficients.

    This is a copy of the implementation found in numpy, with the removal of
    the deperecated bias and ddof keyword arguments, and the addition of
    the fweights and aweights arguments, which are pased to np.cov.

    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `x` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same
        shape as `x`.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    fweights : array_like, int, optional
        1-D array of integer freguency weights; the number of times each
        observation vector should be repeated.
    aweights : array_like, optional
        1-D array of observation vector weights. These relative weights are
        typically large for observations considered "important" and smaller for
        observations considered less "important". If ``ddof=0`` the array of
        weights can be used to assign probabilities to observation vectors.

    Returns
    -------
    R : ndarray
        The correlation coefficient matrix of the variables.
    """

    c = np.cov(x, y, rowvar, fweights=fweights, aweights=aweights)
    try:
        d = np.diag(c)
    except ValueError:
        # scalar covariance
        # nan if incorrect value (nan, inf, 0), 1 otherwise
        return c / c
    stddev = np.sqrt(d.real)
    c /= stddev[:, None]
    c /= stddev[None, :]

    # Clip real and imaginary parts to [-1, 1].  This does not guarantee
    # abs(a[i,j]) <= 1 for complex arrays, but is the best we can do without
    # excessive work.
    np.clip(c.real, -1, 1, out=c.real)
    if np.iscomplexobj(c):
        np.clip(c.imag, -1, 1, out=c.imag)

    return c
