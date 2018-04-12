# -*- coding: utf-8 -*-

"""
This module contains functions which create and train classifiers, as well as
saving them to and reading them from disk.

A classifier function takes a DataFrame containing training data, a list
describing preprocessing steps, and a list of features. It will return a
trained scikit-learn Pipeline containing the preprocessing steps and
classifier.

Classifiers are saved do disk using dill as Python's pickle module does not
correctly serialise Keras classifiers. It should be noted that Keras does not
recommend pickling for neural network serialisation, but no issues have been
observed so far using the dill library.
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
from collections import namedtuple

from sklearn.pipeline import make_pipeline


def evaluate_mva(df, mva):
    """
    Evaluate the response of a trained classifier.

    Parameters
    ----------
    df : DataFrame, shape= [n_training_samples, n_features]
        DataFrame containing features.
    mva
        Trained classifier.

    Returns
    -------
    Series or array
        Classifier response values corresponding to each entry in df.

    Notes
    -----
    The classifier response values are taken from the mva object's
    predict_proba method. By default this is passed the df DataFrame directly
    but in some cases this is not supported and df is passed as a numpy array.
    In the former case this function returns a Pandas Series and in the latter
    a 1D array. This fallback has only been tested for Keras classifiers.
    """

    # Keras doesn't like DataFrames, error thrown depends on Keras version
    try:
        return mva.predict_proba(df)[:, 1]
    except (KeyError, UnboundLocalError):
        return mva.predict_proba(df.as_matrix())[:, 1]


def mlp(df_train, pre, y, serialized_model, sample_weight=None,
        model_params={}, early_stopping_params=None, compile_params={},
        lr_reduction_params=None):
    """
    Train using a multi-layer perceptron (MLP).

    Parameters
    ----------
    df_train : array-like, shape = [n_training_samples, n_features]
        DataFrame containing training features.
    pre : list
        List containing preprocessing steps.
    y : array-like, shape = [n_training_samples]
        Target values (integers in classification, real numbers in regression).
        For classification, labels must correspond to classes.
    serialized_model : dict
        Keras model serialized as a dict.
    sample_weight : array-like, shape = [n_training_samples]
        Sample weights. If None, then samples are equally weighted.
    model_params : dict
        Keyword arguments passed to
        keras.wrappers.scikit_learn.KerasClassifier.
    early_stopping_params : dict
        Keyword arguments passed to keras.callbacks.EarlyStopping. If None, no
        early stopping mechanism is used.
    compile_params : dict
        Keyword arguments passed to keras.models.Sequential.compile.

    Returns
    -------
    Pipeline
        Scikit-learn pipeline containing the trained classifier and
        preprocessing steps.

    Notes
    -----
    This function requires Keras to be available. Additional configuration can
    be configured using Keras' configuration file. See the Keras documentation
    for more information.

    Keras should outperform scikit-learn's internal MLP implementation in most
    cases, and supports sample weights while training.
    """

    def build_model():
        from keras.models import layer_module

        # Set input layer shape
        serialized_model["config"][0]["config"]["batch_input_shape"] \
            = (None, df_train.shape[1])

        model = layer_module.deserialize(serialized_model)

        model.compile(**compile_params)

        return model

    from keras.wrappers.scikit_learn import KerasClassifier

    callbacks = []

    if lr_reduction_params is not None:
        from keras.callbacks import ReduceLROnPlateau
        callbacks.append(ReduceLROnPlateau(**lr_reduction_params))

    if early_stopping_params is not None:
        from keras.callbacks import EarlyStopping
        callbacks.append(EarlyStopping(**early_stopping_params))

    ann = KerasClassifier(build_fn=build_model, **model_params)

    mva = make_pipeline(*(pre + [ann]))

    # Keras does not like pandas
    try:
        df_train = df_train.as_matrix()
    except AttributeError:
        pass
    try:
        y = y.as_matrix()
    except AttributeError:
        pass
    try:
        sample_weight = sample_weight.as_matrix()
    except AttributeError:
        pass

    mva.fit(df_train, y,
            kerasclassifier__sample_weight=sample_weight,
            kerasclassifier__callbacks=callbacks)

    return mva


def bdt_grad(df_train, pre, y, sample_weight=None, **kwargs):
    """
    Train using a gradient boosted decision tree using scikit-learn's
    internal implementation.

    Parameters
    ----------
    df_train : array-like, shape = [n_training_samples, n_features]
        DataFrame containing training features.
    pre : list
        List containing preprocessing steps.
    y : array-like, shape = [n_training_samples]
        Target values (integers in classification, real numbers in regression).
        For classification, labels must correspond to classes.
    sample_weight : array-like, shape = [n_training_samples]
        Sample weights. If None, then samples are equally weighted.
    kwargs : dict
        Additional keyword arguments passed to
        sklearn.ensemble.GradientBoostingClassifier.

    Returns
    -------
    Pipeline
        Scikit-learn pipeline containing the trained classifier and
        preprocessing steps.
    """

    from sklearn.ensemble import GradientBoostingClassifier

    bdt = GradientBoostingClassifier(**kwargs)

    mva = make_pipeline(*(pre + [bdt]))

    mva.fit(df_train, y,
            gradientboostingclassifier__sample_weight=sample_weight)

    return mva


def bdt_xgb(df_train, pre, y, sample_weight=None, **kwargs):
    """
    Train using a gradient boosted decision tree with the XGBoost library.

    Parameters
    ----------
    df_train : array-like, shape = [n_training_samples, n_features]
        DataFrame containing training features.
    pre : list
        List containing preprocessing steps.
    y : array-like, shape = [n_training_samples]
        Target values (integers in classification, real numbers in regression).
        For classification, labels must correspond to classes.
    sample_weight : array-like, shape = [n_training_samples]
        Sample weights. If None, then samples are equally weighted.
    kwargs : dict
        Additional keyword arguments passed to xgboost.XGBClassifier.

    Returns
    -------
    Pipeline
        Scikit-learn pipeline containing the trained classifier and
        preprocessing steps.

    Notes
    -----
    Requires xgboost.
    """

    from xgboost import XGBClassifier

    bdt = XGBClassifier(**kwargs)

    mva = make_pipeline(*(pre + [bdt]))

    mva.fit(df_train, y, xgbclassifier__sample_weight=sample_weight)
            # eval_metric="auc",
            # early_stopping_rounds=50,
            # eval_set=[(df_test, sample_weight)])

    return mva


def bdt_lgbm(df_train, pre, y, sample_weight=None, **kwargs):
    """
    Train using a gradient boosted decision tree with the LightGBM library.

    Parameters
    ----------
    df_train : array-like, shape = [n_training_samples, n_features]
        DataFrame containing training features.
    pre : list
        List containing preprocessing steps.
    y : array-like, shape = [n_training_samples]
        Target values (integers in classification, real numbers in regression).
        For classification, labels must correspond to classes.
    sample_weight : array-like, shape = [n_training_samples]
        Sample weights. If None, then samples are equally weighted.
    kwargs : dict
        Additional keyword arguments passed to lightgbm.LGBMClassifier()

    Returns
    -------
    Pipeline
        Scikit-learn pipeline containing the trained classifier and
        preprocessing steps.

    Notes
    -----
    Requires xgboost.
    """

    from lightgbm import LGBMClassifier

    bdt = LGBMClassifier(**kwargs)

    mva = make_pipeline(*(pre + [bdt]))

    mva.fit(df_train, y, lgbmclassifier__sample_weight=sample_weight)

    return mva


def random_forest(df_train, pre, y, sample_weight=None, **kwargs):
    """
    Train using a random forest.

    Parameters
    ----------
    df_train : array-like, shape = [n_training_samples, n_features]
        DataFrame containing training features.
    pre : list
        List containing preprocessing steps.
    y : array-like, shape = [n_training_samples]
        Target values (integers in classification, real numbers in regression).
        For classification, labels must correspond to classes.
    sample_weight : array-like, shape = [n_training_samples]
        Sample weights. If None, then samples are equally weighted.
    kwargs : dict
        Additional keyword arguments passed to xgboost.XGBClassifier.

    Returns
    -------
    Pipeline
        Scikit-learn pipeline containing the trained classifier and
        preprocessing steps.
    """

    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(**kwargs)

    mva = make_pipeline(*(pre + [rf]))

    mva.fit(df_train, y, randomforestclassifier__sample_weight=sample_weight)

    return mva


def save_classifier(mva, cfg=None, filename="mva"):
    """
    Write a trained classifier pipeline and global configuration to an external
    file.

    Parameters
    ----------
    mva : trained classifier
        Classifier to be trained.
    cfg : dict, optional
        Classifier configuration.
    filename : string, optional
        Name of output file (including directory). Extension will be set
        automatically.

    Returns
    -------
    None

    Notes
    -----
    Requires dill.
    """

    import dill

    SavedClassifier = namedtuple("SavedClassifier", "cfg mva")

    # Temporarily boost the recursion limit
    tmp = sys.getrecursionlimit()
    sys.setrecursionlimit(9999)

    dill.dump(SavedClassifier(cfg, mva), open("{}.pkl".format(filename), "wb"))

    sys.setrecursionlimit(tmp)


def load_classifier(f):
    """
    Load a trained classifier from a pickle file.

    Parameters
    ----------
    f : file
        File classifier is to be loaded from.

    Returns
    -------
    mva: Pipeline
        Scikit-learn Pipeline containing full classifier stack.
    cfg:
        Configuration associated with mva. None if no configuration was
        stored.

    Notes
    -----
    Requires dill.
    """

    import dill

    sc = dill.load(f)

    return sc.mva, sc.cfg
