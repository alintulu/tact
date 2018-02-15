# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import unittest

import numpy as np
import pandas as pd
import ROOT

from context import tact
from tact import metrics

np.random.seed(52)


class ECDFTests(unittest.TestCase):
    """
    Tests for metrics.ecdf
    """

    def setUp(self):
        self.a = np.linspace(0, 1, num=10)
        np.random.shuffle(self.a)
        self.w = np.random.rand(10) - 0.25

    def test_right_handed(self):
        """
        Check the ecdf increases at an observation, not immediately after it.
        As the weight at the first observation is checked, this also tests
        that weights remain associated with the correct observation.
        """
        ecdf = metrics.ecdf(self.a, xw=self.w)
        self.assertAlmostEqual(ecdf(np.min(self.a)),
                               self.w[np.argmin(self.a)] / self.w.sum(),
                               places=14)
        self.assertAlmostEqual(ecdf(1), 1, places=14)

    def test_ecdf_at_zero(self):
        """
        Test the returned ECDF is 0 when below the range of observations.
        """
        ecdf = metrics.ecdf(self.a, xw=self.w)
        self.assertEquals(ecdf(-np.inf), 0)
        self.assertEquals(ecdf(np.nextafter(0, -np.inf)), 0)

    def test_ecdf_at_unity(self):
        """
        Test the returned ECDF is 1 when above the range of observations.
        """
        ecdf = metrics.ecdf(self.a, xw=self.w)
        self.assertAlmostEqual(ecdf(np.inf), 1, places=14)
        self.assertAlmostEqual(ecdf(np.nextafter(1, np.inf)), 1, places=14)

    def test_midpoint(self):
        """
        Test the returned ECDF is 0.5 in the middle of a linearly distributed
        set of unweighted and equally-weighted observations.
        """
        self.assertAlmostEqual(metrics.ecdf(self.a)(0.5), 0.5)
        self.assertAlmostEqual(metrics.ecdf(self.a, xw=np.ones(10) / 8)(0.5),
                               0.5)

    def test_raises_on_zero_normalisation(self):
        """
        Check a ValueError is raised for when weights sum to zero.
        """
        self.assertRaises(ValueError, metrics.ecdf, self.a, np.zeros(1000))
        self.assertRaises(ValueError, metrics.ecdf, self.a,
                          np.concatenate((self.w[:500], -self.w[:500])))

    def test_raises_on_negative(self):
        """
        Check a ValueError is raised for when weights sum to < 0.
        """
        self.assertRaises(ValueError, metrics.ecdf, self.a, -self.w)


class KS2SampTests(unittest.TestCase):
    """
    Tests for metrics.ks_2samp
    """

    def scipy_tests(self):
        """
        ks_2samp tests used by scipy. This implementation should give the same
        answers in the unweighted case.
        """

        from numpy.testing import assert_almost_equal

        # exact small sample solution
        data1 = np.array([1.0, 2.0])
        data2 = np.array([1.0, 2.0, 3.0])
        assert_almost_equal(np.array(metrics.ks_2samp(data1 + 0.01, data2)),
                            np.array((0.33333333333333337, 0.99062316386915694)))
        assert_almost_equal(np.array(metrics.ks_2samp(data1 - 0.01, data2)),
                            np.array((0.66666666666666674, 0.42490954988801982)))
        # these can also be verified graphically
        assert_almost_equal(
            np.array(metrics.ks_2samp(np.linspace(1, 100, 100),
                                      np.linspace(1, 100, 100) + 2 + 0.1)),
            np.array((0.030000000000000027, 0.99999999996005062)))
        assert_almost_equal(
            np.array(metrics.ks_2samp(np.linspace(1, 100, 100),
                                      np.linspace(1, 100, 100) + 2 - 0.1)),
            np.array((0.020000000000000018, 0.99999999999999933)))
        # these are just regression tests
        assert_almost_equal(
            np.array(metrics.ks_2samp(np.linspace(1, 100, 100),
                                      np.linspace(1, 100, 110) + 20.1)),
            np.array((0.21090909090909091, 0.015880386730710221)))
        assert_almost_equal(
            np.array(metrics.ks_2samp(np.linspace(1, 100, 100),
                                      np.linspace(1, 100, 110) + 20 - 0.1)),
            np.array((0.20818181818181825, 0.017981441789762638)))

        def test_series(self):
            """
            Check the same values are returned for pandas Series and numpy
            arrays.
            """
            a = np.random.normal(1, 1, 100)
            b = np.random.normal(1, 1, 100)
            aw = np.random.normal(1, 1, 100)
            bw = np.random.normal(1, 1, 100)

            self.assertEquals(metrics.ks_2samp(a, b, aw, bw),
                              metrics.ks_2samp(pd.Series(a), pd.Series(b),
                                               pd.Series(aw), pd.Series(bw)))

    # TODO create test cases for weighted samples


if __name__ == "__main__":
    unittest.main()
