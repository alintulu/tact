# -*- coding: UTF-8 -*-

from setuptools import setup

setup(name="tact",
      version="0.1pre",
      description="Tool for performing multivariate classificaton on top"
      "analyses",
      url="https://github.com/brunel-physics/tact",
      author="Corin Hoad",
      author_email="c.h@cern.ch",
      license="BSD",
      packages=["tact"],
      entry_points={
          "console_scripts": ["tact=tact.tact:main"
                              "tact2D=tact_2D:main"],
      },
      install_requires=[
          "dill",
          "matplotlib",
          "more_itertools",
          "numpy",
          "pandas",
          "PyYAML"
          "root_numpy",
          "root_pandas",
          "scikit_learn",
          "scipy",
      ],
      extras_require={
          'MLP':  ["Keras"],
          'xgboost': ["xgboost"]
      },
      zip_safe=False)
