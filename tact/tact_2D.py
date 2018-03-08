# -*- coding: utf-8 -*-

"""
2D.py

Usage:
    tact_2D config.yaml
or  tact_2D --stdin < config.yaml
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys

import numpy as np

from tact import classifiers, rootIO, util
from tact import plotting as pt
from tact.config import cfg, read_config

import matplotlib.pyplot as plt

def main():
    # Read configuration
    try:
        read_config()
    except IndexError:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(1)

    np.random.seed(cfg.get("seed"))
    plt.style.use("ggplot")

    # Make ouptut directories
    rootIO.makedirs(cfg["plot_dir"], cfg["root_dir"], cfg["mva_dir"])

    # Load pickled classifiers
    mva1, cfg["mva1"] = classifiers.load_classifier(open(cfg["classifier1"],
                                                         "rb"))
    mva2, cfg["mva2"] = classifiers.load_classifier(open(cfg["classifier2"],
                                                         "rb"))

    if cfg["mva1"] is None or cfg["mva2"] is None:
        print("Error, configuration not saved with one of the provided "
              "classifiers, this is required for usage of this tool.",
              file=sys.stderr)
        sys.exit(2)

    cfg["features"] = list(set(cfg["mva1"]["features"] +
                               cfg["mva2"]["features"]))

    # Read TTrees and evaluate classifiers
    df = rootIO.read_trees(
        cfg["input_dir"], cfg["features"], cfg["signals"], cfg["backgrounds"],
        selection=cfg["selection"],
        negative_weight_treatment=cfg["negative_weight_treatment"],
        equalise_signal=cfg["equalise_signal"])
    df = df.assign(MVA1=classifiers.evaluate_mva(df[cfg["mva1"]["features"]],
                                                 mva1))
    df = df.assign(MVA2=classifiers.evaluate_mva(df[cfg["mva2"]["features"]],
                                                 mva2))

    # Evaluate classifiers
    df = df.assign(MVA1=classifiers.evaluate_mva(df[cfg["mva1"]["features"]],
                                                 mva1))
    df = df.assign(MVA2=classifiers.evaluate_mva(df[cfg["mva2"]["features"]],
                                                 mva2))

    # Plot classifier responses on 2D plane
    process_status = df.Process.map(
        lambda process: ((process in cfg["mva1"]["signals"],
                          process in cfg["mva2"]["signals"]),
                         (process in cfg["mva1"]["backgrounds"],
                          process in cfg["mva2"]["backgrounds"])))

    pt.make_scatter_plot(df["MVA1"], df["MVA2"], s=df["EvtWeight"].abs(),
                         c=process_status.map(hash), cmap="Dark2",
                         marker=",", colorbar=False,
                         filename="{}scatter_{}.pdf"
                         .format(cfg["plot_dir"], cfg["channel"]))

    # Combine classifier scores
    if cfg["combination"] == "min":
        response = lambda x: np.minimum(
            classifiers.evaluate_mva(x[cfg["mva1"]["features"]], mva1),
            classifiers.evaluate_mva(x[cfg["mva2"]["features"]], mva2))
        range = (0, 1)
    elif cfg["combination"] == "max":
        response = lambda x: np.maximum(
            classifiers.evaluate_mva(x[cfg["mva1"]["features"]], mva1),
            classifiers.evaluate_mva(x[cfg["mva2"]["features"]], mva2))
        range = (0, 1)
    elif cfg["combination"] == "add":
        response = lambda x: \
                classifiers.evaluate_mva(x[cfg["mva1"]["features"]], mva1) + \
                classifiers.evaluate_mva(x[cfg["mva2"]["features"]], mva2)
        range = (0, 2)
    elif cfg["combination"] == "quadrature":
        response = lambda x: np.sqrt(
            np.square(
                classifiers.evaluate_mva(x[cfg["mva1"]["features"]], mva1)) +
            np.square(
                classifiers.evaluate_mva(x[cfg["mva2"]["features"]], mva2)))
        range = (0, np.sqrt(2))
    elif cfg["combination"] == "PCA":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=1, svd_solver="full")
        km = pca.fit(df[["MVA1", "MVA2"]])

        response = lambda x: pca.transform(
            np.column_stack((
                classifiers.evaluate_mva(x[cfg["mva1"]["features"]], mva1),
                classifiers.evaluate_mva(x[cfg["mva2"]["features"]], mva2))))

        # Calculate histogram range by examining extreme values
        extremes = pca.transform([[0, 0], [0, 1], [1, 0], [1, 1]])
        range = (min(extremes), max(extremes))
    elif cfg["combination"] == "kmeans":
        from sklearn.cluster import KMeans

        n_clusters = 20

        km = KMeans(n_clusters=n_clusters, n_jobs=-1)
        km = km.fit(df[["MVA1", "MVA2"]])
        df = df.assign(kmean=km.predict(df[["MVA1", "MVA2"]]))

        # Create a lookup table mapping the cluster lables to their ranking in
        # S/N ratio
        clusters = [el[1] for el in df.groupby("kmean")]
        lut = {}
        for i, cluster in enumerate(
                sorted(clusters, key=lambda x:
                       util.s_to_n(x.Signal, x.EvtWeight)), 0):
            lut[cluster.kmean.iloc[0]] = i

        response = lambda x: np.vectorize(lut.__getitem__)(km.predict(
            np.column_stack((
                classifiers.evaluate_mva(x[cfg["mva1"]["features"]], mva1),
                classifiers.evaluate_mva(x[cfg["mva2"]["features"]], mva2)))))

        # Set output range and override the number of bins - cluster label is
        # not continuous.
        range = (0, n_clusters)
        cfg["root_out"]["bins"] = n_clusters

        pt.make_cluster_region_plot(km, cmap="tab20",
                                    filename="{}kmeans_clusters_{}.pdf"
                                    .format(cfg["plot_dir"], cfg["channel"]))
        pt.make_scatter_plot(df.MVA1, df.MVA2, marker=",", c=df.kmean,
                             s=df.EvtWeight.abs(), cmap="tab20",
                             colorbar=False,
                             filename="{}kmeans_areas_{}.pdf"
                             .format(cfg["plot_dir"], cfg["channel"]))
    else:
        raise ValueError("Unrecogised value for option 'combination': ",
                         cfg["combination"])

    rootIO.write_root(cfg["input_dir"], cfg["features"], response,
                      selection=cfg["selection"], bins=cfg["root_out"]["bins"],
                      range=range, combine=cfg["root_out"]["combine"],
                      data=cfg["root_out"]["data"],
                      drop_nan=cfg["root_out"]["drop_nan"],
                      channel=cfg["channel"], data_process=cfg["data_process"],
                      filename="{}mva_{}.root".format(cfg["root_dir"],
                                                      cfg["channel"]))


if __name__ == "__main__":
    main()
