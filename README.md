# Top Analysis Classification Tool (tact)

A program to apply multivariate classification techniques to top analyses.

## Getting Started

### Prerequisites

Requires `python>=2.7.13` and `root>=6.06.08` with python 2 bindings enabled.
Python 3 is not supported.

### Installing

To install run
```bash
pip install -e .
```

For multi-layer perception support, specify:
```bash
pip install -e .[MLP]
```
This will install the Keras package, but not a (required!) Keras backend.
Consult the Keras documentation for availible backends, but note only the
Theano backend has been tested.

For gradient boosted decision trees using the xgboost library, specify:
```bash
pip install -e .[xgboost]
```

### Usage

This tool is used via the `tact` and `tact_2D` command-line utilities. Both
take only one argument - a YAML configuration file, well-documented examples
of which can be found in the "configs/" directory. Alternatively, the `--stdin`
argument can be specified to read a configuration file from stdin.

`tact` performs multivariate classifier training and application, producing
files which can then be used in THETA or the Higgs Analysis Combined Limit
tools. It should be the only program needed for most use-cases.

`tact_2D` performs dimensionality reduction for two classifiers trained and
and saved by the `tact` program.

### Authors
+ Corin Hoad

### License
This project is licensed under the Revised BSD License. See the LICENSE.txt for
details.
