# -*- coding: utf-8 -*-

"""
This module contains functions relating to more advanced binning techniques,
in particular recursive binning algorithms.
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from tact import util


def _meets_num_threshold(xw, cat, s_num_thresh=1, b_num_thresh=1,
                         s_err_thresh=0.3, b_err_thresh=0.3):
    """
    Check if the number of signal and background events in xw are above the
    specified number threshold and above the specified error threshold.

    Parameters
    ----------
    xw : array-like, shape=N
        Event weights.
    cat : 1D array, shape=N
        Array containing labels describing whether an entry is signal (1 or
    True) or background (0 or False).
    s_num_thresh, b_num_thresh, float, optional
        Signal and background event count thresholds.
    s_num_thresh, b_num_thresh, float, optional
        Signal and background event bin error thresholds.

    Returns
    -------
    bool
        True if meets either threshold, False otherwise.
    """

    sums = xw[cat == 1].sum()
    sumb = xw[cat == 0].sum()

    return sumb < b_num_thresh or sums < s_num_thresh \
        or (xw[cat == 0] ** 2).sum() ** 0.5 / sumb > s_err_thresh \
        or (xw[cat == 1] ** 2).sum() ** 0.5 / sums > b_err_thresh


def _recursive_median_tree(x, cat, xw=None, s_num_thresh=1, b_num_thresh=1,
                           s_err_thresh=0.3, b_err_thresh=0.3):
    """
    Perform binning by recursively finding the median.

    The provided data is split at the median. The resulting subsamples are
    continually split until doing so would result in a sample with less than
    s_num_thresh (b_num_thresh) signal (background) events or a % bin error
    greater than s_err_thresh in signal or b_err_thresh in background.

    Parameters
    ----------
    x : array-like, shape=N
        Data to be binned.
    cat : 1D array, shape=N
        Array containing labels describing whether an entry is signal (1 or
        True) or background (0 or False).
    xw : array-like, shape=N, optional
        Weights for samples in x. If None, equal weights are used.
    s_num_thresh, b_num_thresh, float, optional
        Minimum number of samples in a cluster in signal or background before
        splitting is stopped.
    s_err_thresh, b_err_thresh, float, optional
        Maximum percentage error in a cluster in signal or background before
        splitting is stopped.

    Returns
    -------
    BinaryTree
        BinaryTree containing subsample medians.
    """

    if xw is None:
        xw = np.ones(len(x))

    median = np.median(x)
    mask = (x < median)

    if _meets_num_threshold(
            xw[mask], cat[mask], s_num_thresh, b_num_thresh,
            s_err_thresh, b_err_thresh) or \
            _meets_num_threshold(
                xw[~mask], cat[~mask], s_num_thresh, b_num_thresh,
                s_err_thresh, b_err_thresh):
        return None

    tree = util.BinaryTree()
    tree.val = median
    tree.left = _recursive_median_tree(x[mask], cat[mask], xw[mask],
                                       s_num_thresh, b_num_thresh,
                                       s_err_thresh, b_err_thresh)
    tree.right = _recursive_median_tree(x[~mask], cat[~mask], xw[~mask],
                                        s_num_thresh, b_num_thresh,
                                        s_err_thresh, b_err_thresh)

    return tree


def recursive_median(x, cat, xw=None, s_num_thresh=1, b_num_thresh=1,
                     s_err_thresh=0.3, b_err_thresh=0.3):
    """
    Perform binning by recursively finding the median.

    The provided data is split at the median. The resulting subsamples are
    continually split until doing so would result in a sample with less than
    s_num_thresh (b_num_thresh) signal (background) events or a % bin error
    greater than s_err_thresh in signal or b_err_thresh in background.

    Parameters
    ----------
    x : array-like, shape=N
        Data to be binned.
    cat : 1D array, shape=N
        Array containing labels describing whether an entry is signal (1 or
        True) or background (0 or False).
    xw : array-like, shape=N, optional
        Weights for samples in x. If None, equal weights are used.
    s_num_thresh, b_num_thresh, float, optional
        Minimum number of samples in a bin in signal or background before
        splitting is stopped.
    s_err_thresh, b_err_thresh, float, optional
        Maximum percentage error in a bin in signal or background before
        splitting is stopped.

    Returns
    -------
    bins : array
        Array of bin edges, includes leftmost bin edge at min(x) and
        rightmost bin edge at max(x).
    """

    tree = _recursive_median_tree(x, cat, xw, s_num_thresh, b_num_thresh,
                                  s_err_thresh, b_err_thresh)

    bins = ([np.min(x)] +
            sorted(m for m in util.nodes(tree) if m is not None) +
            [np.max(x)])
    return bins


def _recursive_kmeans_tree(x, cat, xw=None, s_num_thresh=1, b_num_thresh=1,
                           s_err_thresh=0.3, b_err_thresh=0.3, **kwargs):
    """
    Perform clustering using a recursive k-means algorithm.

    The provided data is split into two clusters using k-means with two
    centroids. These are then sub-clustered until a further split would result
    in a population of signal or background not meeting the specified
    thresholds.

    Parameters
    ----------
    x : array-like, shape=[n_samples, n_features]
        Data to be clustered.
    cat : 1D array, shape=n_samples
        Array containing labels describing whether an entry is signal (1 or
        True) or background (0 or False).
    xw : array-like, shape=n_samples, optional
        Weights for samples in x. If None, equal weights are used.
    s_num_thresh, b_num_thresh, float, optional
        Minimum number of samples in a cluster in signal or background before
        splitting is stopped.
    s_err_thresh, b_err_thresh, float, optional
        Maximum percentage error in a bin in signal or background before
        splitting is stopped.
    kwargs
        Additional keyword arguments passed to sklearn.cluster.KMeans

    Returns
    -------
    kmtree : BinaryTree
        BinaryTree containing trained k-means clusterers.
    """

    from sklearn.cluster import KMeans

    tree = util.BinaryTree()
    if x is None:
        xw = np.ones(len(x))

    km = KMeans(n_clusters=2, **kwargs)
    km.fit(x)
    mask = (km.predict(x) == 0)

    if _meets_num_threshold(
            xw[mask], cat[mask], s_num_thresh, b_num_thresh,
            s_err_thresh, b_err_thresh) or \
            _meets_num_threshold(
                xw[~mask], cat[~mask], s_num_thresh, b_num_thresh,
                s_err_thresh, b_err_thresh):
        return None

    tree.val = km
    tree.left = _recursive_kmeans_tree(x[mask], cat[mask], xw[mask],
                                       s_num_thresh, b_num_thresh,
                                       s_err_thresh, b_err_thresh)
    tree.right = _recursive_kmeans_tree(x[~mask], cat[~mask], xw[~mask],
                                        s_num_thresh, b_num_thresh,
                                        s_err_thresh, b_err_thresh)

    return tree


def kmeans_bin_edges(tree):
    """
    For a tree of trained 1D k-means classifiers, retrieve the values where two
    clusters meet.

    Parameters
    ----------
    tree : BinaryTree
        BinaryTree containing 1D k-means classifiers.

    Returns
    -------
    bin_edges:
        Cluster edges. Binning using these values as edges will be equivalent
        to clustering.
    """

    return np.fromiter(sorted(np.mean(km.cluster_centers_)
                              for km in util.nodes(tree) if km is not None),
                       np.float)


def recursive_kmeans(x, cat, xw=None, s_num_thresh=1, b_num_thresh=1,
                     s_err_thresh=0.3, b_err_thresh=0.3, bin_edges=False,
                     **kwargs):
    """
    Perform clustering using a recursive k-means algorithm.

    The provided data is split into two clusters using k-means with two
    centroids. These are then sub-clustered until a further split would result
    in a population of signal or background not meeting the specified
    thresholds.

    Parameters
    ----------
    x : array-like, shape=[n_samples, n_features]
        Data to be clustered.
    cat : 1D array, shape=n_samples
        Array containing labels describing whether an entry is signal (1 or
        True) or background (0 or False).
    xw : array-like, shape=n_samples, optional
        Weights for samples in x. If None, equal weights are used.
    s_num_thresh, b_num_thresh, float, optional
        Minimum number of samples in a cluster in signal or background before
        splitting is stopped.
    s_err_thresh, b_err_thresh, float, optional
        Maximum percentage error in a bin in signal or background before
        splitting is stopped.
    kwargs
        Additional keyword arguments passed to sklearn.cluster.KMeans

    Returns
    -------
    kmtree : BinaryTree
        BinaryTree containing trained k-means clusterers.
    bin_edges:
        If n_features == 1 and bin_edges == true, this contains the values of
        the bin edges for a 1D histogram which will split x by cluster.
        Leftmost and rightmost edges are set to min(x) and max(x) respectively.
    """

    kmtree = _recursive_kmeans_tree(x, cat, xw, s_num_thresh, b_num_thresh,
                                    s_err_thresh, b_err_thresh, **kwargs)

    if bin_edges:
        return kmtree, np.concatenate(([x.min()],
                                       kmeans_bin_edges(kmtree),
                                       [x.max()]))
    else:
        return kmtree


def predict_kmeans_tree(tree, X):
    """
    Predict cluster membership for each sample in X according to a previously
    trained recursive k-means clusterer.

    Parameters
    ----------
    tree : BinaryTree
        BinaryTree containing k-means classifiers.
    X : array-like, shape=[n_samples, n_features]
        Data to run prediction on.

    Returns
    -------
    labels : array, shape=n_samples
        Integer labels describing the cluster each sample belongs to.

    Notes
    -----
    The provided labels are guaranteed to be consistent and unique for each
    cluster, but not consecutive.
    """

    def predict_kmeans_tree_event(t, x):
        """
        Predict cluster membership for a single sample x according to a
        previously trained recursive k-means clusterer.

        Parameters
        ----------
        t : BinaryTree
            BinaryTree containing k-means classifiers.
        x : shape=n_features
            Data to run prediction on.

        Returns
        -------
        label : array, shape=n_samples
            Integer label describing the cluster the sample belongs to.

        Notes
        -----
        The label of a sample is initially 0. It is increased by 2^d every time
        it takes the right path from a node, where d is the depth of that node.
        """

        depth = 0
        cluster = 0
        while t is not None:
            branch = t.val.predict(x)
            cluster += branch * 2 ** depth

            depth += 1

            if branch == 0:
                t = t.left
            elif branch == 1:
                t = t.right

        return cluster

    return np.vectorize(predict_kmeans_tree_event)(tree, X)
