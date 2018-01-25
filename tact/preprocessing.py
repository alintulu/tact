# -*- coding: utf-8 -*-

"""
This modules contain functions which add scikit-learn preprocessors to lists,
which can later be transformed into a scikit-learn Pipeline.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


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

    from sklearn.preprocessing import StandardScaler

    return l.append(StandardScaler(**kwargs))


def add_robust_scaler(l, **kwargs):
    """
    Appends a scikit-learn RobustScaler to l.

    Parameters
    ----------
    l : list
        Pipeline to be modified
    kwargs : keyword arguments
        Keyword arguments to be passed to sklearn.preprocessing.RobustScaler.

    Returns
    -------
    list
        Modified list
    """

    from sklearn.preprocessing import RobustScaler

    return l.append(RobustScaler(**kwargs))
