# -*- coding: utf-8 -*-

"""
This modules contain functions which add scikit-learn preprocessors to lists,
which can later be transformed into a scikit-learn Pipeline.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler

class StandardScalerW(StandardScaler):

    def fit(self, X, y, sample_weight=None):

        if sample_weight is None:
            return super(StandardScalerW, self).fit(X, y)

        if sparse.issparse(X):
            raise ValueError("Sparse matrix not supported")

        self._reset()

        print(sample_weight)
        average = np.average(X, axis=0, weights=sample_weight)

        if self.with_mean:
            self.mean_ = average
        if self.with_std:
            from sklearn.preprocessing.data import _handle_zeros_in_scale

            self.var_ = [np.cov(row, aweights=np.abs(sample_weight))
                         for row in X.T]
            self.scale_ = _handle_zeros_in_scale(np.sqrt(self.var_))
        else:
            self.scale_ = None

        return self


def add_standard_scaler(l, **kwargs):
    """
    Appends a scikit-learn StandardScaler to l.

    Parameters
    ----------
    l : list
        Pipeline to be modified
    kwargs : keyword arguments
        Keyword arguments to be passed to sklearn.preprocessing.StandardScaler.

    Returns
    -------
    list
        Modified list
    """

    return l.append(StandardScalerW(**kwargs))


def add_PCA(l, **kwargs):
    """
    Appends a scikit-learn PCA decompositor to l.

    Parameters
    ----------
    l : list
        Pipeline to be modified
    kwargs : keyword arguments
        Keyword arguments to be passed to RobustScaler()

    Returns
    -------
    list
        Modified list
    """

    from sklearn.decomposition import PCA

    return l.append(PCA(**kwargs))
