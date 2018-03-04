# -*- coding: utf-8 -*-

"""
This module contains functions relating to more advanced binning techniques,
in particular recursive binning algorithms.
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from tact.util import BinaryTree


def recursive_kmeans(x, cat, xw=None, s_thresh=1, b_thresh=1, **kwargs):
    """
    Perform clustering using a recursive k-means algorithm.

    The provided data is split into two clusters using k-means with two
    centroids. These are then sub-clustered until a further split would result
    in a population of signal or background below the specified thresholds.

    Parameters
    ----------
    x : array-like, shape=[n_samples, n_features]
        Data to be clustered.
    cat : 1D array, shape=N
        Array containing labels describing whether an entry is signal (1 or
        True) or background (0 or False).
    xw : array-like, shape=n_samples, optional
        Weights for samples in x. If none, equal weights are used.
    s_thresh, b_thresh, float, optional
        Minimum number of samples in a cluster in signal or background before
        splitting is stopped.
    kwargs
        Additional keyword arguments passed to sklearn.cluster.KMeans

    Returns
    -------
    kmtree : BinaryTree
        BinaryTree containing trained k-means clusterers.
    """

    from sklearn.cluster import KMeans

    tree = BinaryTree()
    if x is None:
        xw = np.ones(len(x))

    km = KMeans(n_clusters=2, **kwargs)
    km.fit(x.reshape(-1, 1))
    mask = (km.predict(x.reshape(-1, 1)) == 0)

    if xw[mask & (cat == 0)].sum() < b_thresh or \
            xw[mask & (cat == 1)].sum() < s_thresh or \
            xw[~mask & (cat == 0)].sum() < b_thresh or \
            xw[~mask & (cat == 1)].sum() < s_thresh:
        return None

    tree.val = km
    tree.left = recursive_kmeans(x[mask], cat[mask],
                                 xw[mask], s_thresh, b_thresh)
    tree.right = recursive_kmeans(x[~mask], cat[~mask],
                                  xw[~mask], s_thresh, b_thresh)

    return tree


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
