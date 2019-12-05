# -*- coding: utf-8 -*-

"""
This module contains functions which create plots.

These deal only with plots created with matplotlib. Functions handling the
creation of ROOT histograms are found in the rootIO module.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from operator import sub
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import auc, roc_curve
from tact import binning
from tact.rootIO import makedirs
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
        axis.set_xlabel(axis.get_title(), family="monospace", size="xx-large")
        axis.set_ylim(bottom=0)
        axis.set_title("")
        axis.legend(["Signal", "Background"], fontsize="large",
                    frameon=True,
                    fancybox=True,
                    facecolor='w',
                    framealpha=0.8)
        axis.tick_params(axis='x', labelsize="large")
        axis.set_yticklabels([])

    fig.savefig(filename)

    # # Save plots individually
    dir = re.search(r"((?:[^\/]*\/)*)(?:.*)", filename).group(1) or ""
    makedirs(dir + "/features/")
    for axis in ax:
        extent = axis.get_tightbbox(fig.canvas.get_renderer(), True).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig("{}/features/{}.pdf".format(dir, axis.get_xlabel()),
                    bbox_inches=extent)


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
    plt.xticks(xrange(nvars), df.columns, rotation=90, size=15,
               family="monospace")
    ax.yaxis.set_ticks_position("right")
    plt.yticks(xrange(nvars), df.columns, size=15,
               family="monospace")
    ax.tick_params(axis='both', which='both', length=0)  # hide ticks
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    plt.tight_layout()

    fig.savefig(filename, pad_inches=0, bbox_inches="tight")


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
                              label="Signal (test set)")
    ax = x_test_bkg.plot.hist(bins=bins, ax=ax, weights=w_test_bkg,
                              normed=True, range=x_range, alpha=0.5,
                              label="Background (test set)")

    plt.gca().set_prop_cycle(None)  # use the same colours again

    # Plot error bar plots of training samples
    for x, label, w in ((x_train_sig, "Signal (training set)",
                         w_train_sig),
                        (x_train_bkg, "Background (training set)",
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

    ax.tick_params(axis='x', which='both', labelsize="x-large")
    ax.legend(fontsize="small",
              frameon=True,
              fancybox=True,
              facecolor='w',
              framealpha=0.8)
    ax.set_xlabel("Response", fontsize="xx-large")
    ax.set_ylabel("")
    ax.set_ylim(bottom=0)
    ax.set_yticklabels([])

    fig.savefig(filename, pad_inches=0, bbox_inches="tight")


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
                label="ROC curve for {} set (AUROC = {:0.2f})"
                .format(i, roc_auc[i]))

    ax.plot([0, 1], [0, 1], "k--")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("False Positive Rate", fontsize="xx-large")
    ax.set_ylabel("True Positive Rate", fontsize="xx-large")
    ax.tick_params(axis='both', which='both', labelsize="x-large")
    ax.legend(loc="lower right",
              frameon=True,
              fancybox=True,
              facecolor='w',
              framealpha=0.8,
              fontsize="large")

    fig.savefig(filename, pad_inches=0, bbox_inches="tight")
