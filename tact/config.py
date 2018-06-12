# -*- coding: utf-8 -*-

"""
This module parses configuration files.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
from os.path import expanduser

from tact.util import deep_update
from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

# Defaults
cfg = {"seed": None,
       "selection": "",
       "channel": "all",
       "data_process": None,
       "plot_dir": "plots/",
       "root_dir": "root/",
       "mva_dir": "mva/",
       "test_fraction": 0.5,
       "equalise_signal": True,
       "negative_weight_treatment": "passthrough",
       "bdt_grad": {},
       "bdt_xgb": {},
       "bdt_lgbm": {},
       "mlp": {"model_params": {},
               "compile_params": {},
               "early_stopping_params": {},
               "lr_reduction_params": {}},
       "random_forest": {},
       "preprocessors": (),
       "root_out": {"strategy": "equal",
                    "combine": True,
                    "drop_nan": False,
                    "data": "empty",
                    "bins": 20,
                    "min_signal_events": 1,
                    "min_background_events": 1,
                    "max_signal_error": 0.3,
                    "max_background_error": 0.3},
       }


def read_config():
    """
    Read the configuration file supplied at the command line, or listen on
    stdin.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    global cfg

    if sys.argv[1] == "--stdin":
        f = sys.stdin
    else:
        f = open(sys.argv[1], 'r')

    cfg = deep_update(cfg, load(f, Loader=Loader))

    # Expand paths
    for path_var in ("input_dir", "plot_dir", "mva_dir", "root_dir"):
        try:
            cfg[path_var] = expanduser(cfg[path_var])
        except IndexError:
            pass
