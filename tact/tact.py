# -*- coding: utf-8 -*-

"""
mva_analysis.py

Usage:
    tact config.yaml
or  tact --stdin < config.yaml
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tact import plotting as pt
from tact import classifiers, metrics, preprocessing, rootIO, util, binning
from tact.config import cfg, read_config


def main():
    # Read configuration
    try:
        read_config()
    except IndexError:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(1)

    np.random.seed(cfg["seed"])
    plt.style.use("ggplot")

    # Make ouptut directories
    rootIO.makedirs(cfg["plot_dir"], cfg["root_dir"], cfg["mva_dir"])

    # Read samples
    df = rootIO.read_trees(
        cfg["input_dir"], cfg["features"], cfg["signals"], cfg["backgrounds"],
        selection=cfg["selection"],
        negative_weight_treatment=cfg["negative_weight_treatment"],
        equalise_signal=cfg["equalise_signal"])

    features = cfg["features"]

    # Configure preprocessing
    pre = []
    for p in cfg["preprocessors"]:
        if p["preprocessor"] == "robust_scaler":
            preprocessing.add_robust_scaler(pre, **p["config"])
        elif p["preprocessor"] == "standard_scaler":
            preprocessing.add_standard_scaler(pre, **p["config"])
        elif p["preprocessor"] == "PCA":
            preprocessing.add_PCA(pre, **p["config"])

    # Make plots
    sig_df = df[df.Signal == 1]
    bkg_df = df[df.Signal == 0]

    pt.make_variable_histograms(sig_df[features], bkg_df[features], bins=42,
                                filename="{}vars_{}.pdf"
                                .format(cfg["plot_dir"], cfg["channel"]))
    pt.make_corelation_plot(sig_df[features],
                            filename="{}corr_sig_{}.pdf"
                            .format(cfg["plot_dir"], cfg["channel"]))
    pt.make_corelation_plot(bkg_df[features],
                            filename="{}corr_bkg_{}.pdf"
                            .format(cfg["plot_dir"], cfg["channel"]))

    # Split sample
    df_train, df_test = train_test_split(df, test_size=cfg["test_fraction"])

    # Classify
    if cfg["classifier"] == "mlp":
        mva = classifiers.mlp(
            df_train[features], pre, df_train.Signal, cfg["mlp"]["model"],
            sample_weight=df_train.MVAWeight,
            model_params=cfg["mlp"]["model_params"],
            early_stopping_params=cfg["mlp"].get("early_stopping_params"),
            compile_params=cfg["mlp"]["compile_params"],
            lr_reduction_params=cfg["mlp"]["lr_reduction_params"])
    elif cfg["classifier"] == "bdt_ada":
        mva = classifiers.bdt_ada(df_train[features], pre, df_train.Signal,
                                  sample_weight=df_train.MVAWeight,
                                  **cfg["bdt_ada"])
    elif cfg["classifier"] == "bdt_xgb":
        mva = classifiers.bdt_xgb(df_train[features], pre, df_train.Signal,
                                  sample_weight=df_train.MVAWeight,
                                  **cfg["bdt_xgb"])
    elif cfg["classifier"] == "bdt_grad":
        mva = classifiers.bdt_grad(df_train[features], pre, df_train.Signal,
                                   sample_weight=df_train.MVAWeight,
                                   **cfg["bdt_grad"])
    elif cfg["classifier"] == "random_forest":
        mva = classifiers.random_forest(df_train[features], pre,
                                        df_train.Signal,
                                        sample_weight=df_train.MVAWeight,
                                        **cfg["random_forest"])
    elif cfg["classifier"] == "load":
        mva = classifiers.load_classifier(open(cfg["classifier_path"]))[0]
    else:
        raise ValueError("Unrecognised value for option 'classifier': ",
                         cfg["classifier"])

    df_test = df_test.assign(MVA=classifiers.evaluate_mva(df_test[features],
                                                          mva))
    df_train = df_train.assign(MVA=classifiers.evaluate_mva(df_train[features],
                                                            mva))
    df = df.assign(MVA=pd.concat((df_train.MVA, df_test.MVA)))

    # Save trained classifier
    classifiers.save_classifier(mva, cfg, "{}{}_{}".format(cfg["mva_dir"],
                                                           cfg["classifier"],
                                                           cfg["channel"]))

    # Metrics
    metrics.print_metrics(mva, df_train[features], df_test[features],
                          df_train.Signal, df_test.Signal,
                          df_train.MVA, df_test.MVA,
                          df_train.EvtWeight, df_test.EvtWeight)

    pt.make_response_plot(df_train[df_train.Signal == 1].MVA,
                          df_test[df_test.Signal == 1].MVA,
                          df_train[df_train.Signal == 0].MVA,
                          df_test[df_test.Signal == 0].MVA,
                          df_train[df_train.Signal == 1].EvtWeight,
                          df_test[df_test.Signal == 1].EvtWeight,
                          df_train[df_train.Signal == 0].EvtWeight,
                          df_test[df_test.Signal == 0].EvtWeight,
                          filename="{}response_{}.pdf".format(cfg["plot_dir"],
                                                              cfg["channel"]))
    pt.make_roc_curve(df_train.MVA, df_test.MVA,
                      df_train.Signal, df_test.Signal,
                      df_train.EvtWeight, df_test.EvtWeight,
                      filename="{}roc_{}.pdf".format(cfg["plot_dir"],
                                                     cfg["channel"]))


    # Binning
    response = lambda x: classifiers.evaluate_mva(x[features], mva)
    outrange = (0, 1)

    if cfg["root_out"].get("binning_strategy") == "equal" or \
            cfg["root_out"].get("binning_strategy") is None:
        bins = cfg["root_out"]["bins"]
    elif cfg["root_out"]["binning_strategy"] == "quantile":
        bins = df.MVA.quantile(np.linspace(0, 1, cfg["root_out"]["bins"] + 1))
        bins[0] = outrange[0]
        bins[-1] = outrange[1]
    elif cfg["root_out"]["binning_strategy"] == "recursive_median":
        bins = binning.recursive_median(df.MVA, df.Signal, df.EvtWeight,
                                        s_thresh=1, b_thresh=1)
        bins[0] = outrange[0]
        bins[-1] = outrange[1]
    elif cfg["root_out"]["binning_strategy"] == "recursive_kmeans":
        _, bins = binning.recursive_kmeans(
            df.MVA, df.Signal, xw=df.EvtWeight,
            s_thresh=cfg["root_out"]["min_signal_events"],
            b_thresh=cfg["root_out"]["min_background_events"],
            bin_edges=True, n_jobs=-1)
        bins[0] = outrange[0]
        bins[-1] = outrange[1]
    else:
        raise ValueError("Unrecognised value for option 'binning_strategy': ",
                         cfg["root_out"]["binning_strategy"])

    rootIO.write_root(
        cfg["input_dir"], cfg["features"], response,
        selection=cfg["selection"], bins=bins,
        data=cfg["root_out"]["data"], combine=cfg["root_out"]["combine"],
        data_process=cfg["data_process"], drop_nan=cfg["root_out"]["drop_nan"],
        channel=cfg["channel"], range=outrange,
        filename="{}mva_{}.root".format(cfg["root_dir"], cfg["channel"]))


if __name__ == "__main__":
    main()
