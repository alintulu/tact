# -*- coding: utf-8 -*-

"""
This module contains functions which create plots.

These deal only with plots created with matplotlib. Functions handling the
creation of ROOT histograms are found in the rootIO module.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from operator import sub

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import auc, roc_curve
from tact import binning
from tact.util import BinaryTree, corrcoef, maenumerate


def make_variable_histograms(df, cat, w=None, filename="vars.pdf", **kwargs):
    """
    Produce histograms comparing the distribution of data in df_sig and df_bkg.

    Histograms are produced for every column in the provided DataFrames.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing data.
    cat : 1D array, shape=N
        Array containing labels describing whether an entry is signal (1 or
        True) or background (0 or False).
    w : array-like, shape=N
        Weights for data. If None, then samples are equally weighted.
    filename : string, optional
        Name of the file the plot is saved to.
    kwargs :
        Additional kwargs passed to pandas.DataFrame.hist

    Returns
    -------
    None
    """

    if w is None:
        w = np.ones(len(df))

    mask = (cat == 1)
    df_sig = df[mask]
    df_bkg = df[~mask]
    w_sig = w[mask]
    w_bkg = w[~mask]

    def plot_histograms(df, ax, w=None):
        """Plot histograms for every column in df"""
        return df.hist(ax=ax, alpha=0.5, weights=w, normed=True, **kwargs)

    n_histograms = len(df_sig.columns)

    ncols = 2
    nrows = (n_histograms + ncols - 1) // ncols

    fig_size = (ncols * 1.618 * 3, nrows * 3)

    fig, ax = plt.subplots(ncols=ncols, nrows=nrows)
    fig.set_size_inches(fig_size)

    ax = ax.flatten()

    for i in xrange(1, n_histograms % ncols + 1):
        ax[-i].remove()

    ax = ax[:n_histograms]

    ax = plot_histograms(df_sig, ax, w_sig)
    plot_histograms(df_bkg, ax, w_bkg)

    for axis in ax:
        axis.legend(["Signal", "Background"], fontsize="x-small")

    fig.savefig(filename)


def make_corelation_plot(df, w=None, filename="corr.pdf", **kwargs):
    """
    Produce matshow plot representing the correlation matrix of df.

    Correlation coefficients are calculated between every column in df.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing data for which the correlation coefficients are to
        be calculated.
    filename : string, optional
        Name of the file the plot is saved to.
    kwargs
        Additional kwargs passed to matplotlib.pyplot.matshow

    Returns
    -------
    None
    """

    corr = corrcoef(df.values, rowvar=False, aweights=np.abs(w))

    nvars = len(df.columns)

    fig, ax = plt.subplots()

    corr_masked = np.ma.array(corr,
                              mask=np.tri(corr.shape[0], k=-1, dtype=np.bool))
    cmap = matplotlib.cm.bwr
    cmap.set_bad('white', 1.)
    ax.matshow(corr_masked, vmin=-1, vmax=1, cmap=cmap, **kwargs)

    for (i, j), z in maenumerate(corr_masked):
        ax.text(j, i, '{}'.format(int(round(z * 100))),
                ha='center', va='center', fontsize=15,
                bbox=dict(boxstyle='square', facecolor='white', alpha=0.8,
                          lw=0))

    fig.set_size_inches(1 + nvars / 1.5, 1 + nvars / 1.5)
    plt.xticks(xrange(nvars), df.columns, rotation=90, fontsize=15)
    ax.yaxis.set_ticks_position("right")
    plt.yticks(xrange(nvars), df.columns, fontsize=15)
    ax.tick_params(axis='both', which='both', length=0)  # hide ticks
    ax.grid(False)

    plt.tight_layout()

    fig.savefig(filename)


def make_response_plot(x_train_sig, x_test_sig, x_train_bkg, x_test_bkg,
                       w_train_sig=None, w_test_sig=None,
                       w_train_bkg=None, w_test_bkg=None,
                       bins=25, filename="response.pdf"):
    """
    Produce histogram comparing the response of the test data and training data
    in signal and background.

    Typically used to compare the distribution of the MVA response.

    Parameters
    ----------
    x_train_sig : Series
        Series containing signal training data.
    x_test_sig : Series
        Series containing signal testing data.
    x_train_bkg : Series
        Series containing background training data.
    x_test_bkg : Series
        Series containing background testing data.
    w* : array-like
        Weights for the corresponding series. If None, then samples are
        equally weighted.
    bins : int, optional
        Number of bins in histogram.
    filename : string, optional
        Name of the file the plot is saved to.
    """

    x_range = (0, 1)

    fig, ax = plt.subplots()

    # Plot histograms of test samples
    ax = x_test_sig.plot.hist(bins=bins, ax=ax, weights=w_test_sig,
                              normed=True, range=x_range, alpha=0.5,
                              label="Signal (test sample)")
    ax = x_test_bkg.plot.hist(bins=bins, ax=ax, weights=w_test_bkg,
                              normed=True, range=x_range, alpha=0.5,
                              label="Background (test sample)")

    plt.gca().set_prop_cycle(None)  # use the same colours again

    # Plot error bar plots of training samples
    for x, label, w in ((x_train_sig, "Signal (training sample)",
                         w_train_sig),
                        (x_train_bkg, "Background (training sample)",
                         w_train_bkg)):
        hist, bin_edges = np.histogram(x, bins=bins, range=x_range,
                                       weights=w)
        hist2 = np.histogram(x, bins=bins, range=x_range,
                             weights=w ** 2)[0]
        db = np.array(np.diff(bin_edges), float)
        yerr = np.sqrt(hist2) / db / hist.sum()
        hist = hist / db / hist.sum()
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        ax.errorbar(bin_centers, hist, fmt=",", label=label,
                    yerr=yerr, xerr=(-sub(*x_range) / bins / 2))

    ax.legend(fontsize="small")

    fig.savefig(filename)


def make_roc_curve(mva_response_train, mva_response_test, y_train, y_test,
                   w_train=None, w_test=None, filename="roc.pdf"):
    """
    Plot the receiver operating characteristic curve for the test and training
    data.

    Parameters
    ----------
    mva_response_train : array-like, shape = [n_training_samples]
        Series containing classifier responses for training data.
    mva_response_test : array-like, shape = [n_testing_samples]
        Series containing classifier responses for test data.
    y_train : array-like, shape = [n_training_samples]
        Series containing target values for training data.
    y_test : array-like, shape = [n_testing_samples]
        Series containing target values for test data.
    filename : string, optional
        Name of the file the plot is saved to.
    w_train : array-like, shape = [n_training_samples], optional
        Weights for df_train. If None, then samples are equally
        weighted.
    w_test : array-like, shape = [n_testing_samples], optional
        Weights for df_test. If None, then samples are equally weighted.

    Returns
    -------
    None
    """

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i, x in (("train", {"response": mva_response_train,
                            "target": y_train,
                            "w": w_train}),
                 ("test", {"response": mva_response_test,
                           "target": y_test,
                           "w": w_test})):
        fpr[i], tpr[i], _ = roc_curve(x["target"], x["response"],
                                      sample_weight=x["w"])
        roc_auc[i] = auc(fpr[i], tpr[i], reorder=True)

    fig, ax = plt.subplots()

    for i in fpr:
        ax.plot(fpr[i], tpr[i],
                label="ROC curve for {} set (auc = {:0.2f})"
                .format(i, roc_auc[i]))

    ax.plot([0, 1], [0, 1], "k--")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    fig.savefig(filename)


def make_scatter_plot(x, y, filename="scatter.pdf", **kwargs):
    """
    Plot the responses of two classifiers for every item in df on a scatter
    plot.

    Events will be coloured red, if they are signal in the first
    classifier, blue if they are background in the first classifier, and green
    otherwise.

    The size of points will be scaled according to the absolute value of the
    associated weight.

    Parameters
    ----------
    x : Series
        Series containing x positions of observations
    y : Series
        Series containing y positions of observations
    filename : string, optional
        Name of the file the plot is saved to.
    kwargs
        Keyword arguments passed to pandas.Dataframe.plot.scatter()

    Returns
    -------
    None
    """

    fig, ax = plt.subplots()

    df = pd.concat([x, y], axis=1, copy=False)

    df.plot.scatter(x.name, y.name, ax=ax, **kwargs)

    fig.savefig(filename)


def make_cluster_region_plot(c, filename="kmeans_areas.pdf", **kwargs):
    """
    Plot the result of k-means clustering. This produces a scatter plot similar
    to make_scatter_plot but colour-coded by cluster and a plot showing
    the extent of each cluster on the 2D plane.

    Parameters
    ----------
    c
        Trained clusterer. Must implement .predict().
    df : DataFrame
        DataFrame containing data to be displayed.
    x : Series
        Series containing x positions of observations
    y : Series
        Series containing y positions of observations
    filename : string, optional
        Name of the file the plot is saved to.
    kwargs :
        Keyword arguments passed to matplotlib.pyplot.imshow

    Returns
    -------
    None
    """

    # First plot shows the full extent of each cluster
    fig, ax = plt.subplots()

    x_min = 0
    x_max = 1
    y_min = 0
    y_max = 1
    h = (x_max - x_min) / 1000
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    if isinstance(c, BinaryTree):
        Z = binning.predict_kmeans_tree(c, (np.c_[xx.ravel(), yy.ravel()]))
    else:
        Z = c.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.imshow(Z, interpolation='nearest',
              extent=(xx.min(), xx.max(), yy.min(), yy.max()),
              aspect='auto', origin='lower', **kwargs)

    ax.grid(False)

    fig.savefig(filename)
