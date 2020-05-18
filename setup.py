# -*- coding: UTF-8 -*-

from setuptools import setup, find_packages

setup(name="tact",
      version="0.1pre",
      description="Tool for performing multivariate classificaton on top"
      "analyses",
      url="https://github.com/brunel-physics/tact",
      author="Corin Hoad",
      author_email="c.h@cern.ch",
      license="BSD",
      packages=find_packages(),
      entry_points={
          "console_scripts": ["tact=tact.tact:main"],
      },
      install_requires=[
          "dill",
          "matplotlib",
          "numpy",
          "pandas",
          "PyYAML",
          "root_numpy",
          "root_pandas",
          "scikit_learn",
          "scipy",
      ],
      extras_require={
          'MLP':  ["Keras"],
          'xgboost': ["xgboost"],
          'lightgbm': ["lightgbm"]
      },
      zip_safe=False)
