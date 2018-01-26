# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import unittest

import numpy as np
import ROOT

from context import tact
from tact import rootIO

np.random.seed(52)


class BalanceWeightsTests(unittest.TestCase):
    """
    Tests for rootIO.balance_weights
    """

    def setUp(self):
        self.a = np.random.rand(1000)
        self.b = np.random.rand(500)

    def test_weights_are_scaled_up(self):
        """
        Ensure weights are always scaled up, not down.
        """
        a_sum = np.sum(self.a)
        b_sum = np.sum(self.b)
        w1, w2 = rootIO.balance_weights(self.a, self.b)
        self.assertTrue(np.sum(w1) >= a_sum)
        self.assertTrue(np.sum(w2) >= b_sum)

    def test_weights_are_balanced(self):
        """
        Check weights are balanced.
        """
        w1, w2 = rootIO.balance_weights(self.a, self.b)
        self.assertTrue(np.isclose(np.sum(w1), np.sum(w2)))

    def test_negative_weights_are_balanced(self):
        """
        Check negative weights are balanced.
        """
        w1, w2 = rootIO.balance_weights(self.a - 1, self.b - 1)
        self.assertTrue(np.isclose(np.sum(w1), np.sum(w2)))

    def test_raises_on_zeroes(self):
        """
        Check that ValueError is raised if normalisation is zero.
        """
        z = np.zeros(1000)
        self.assertRaises(ValueError, rootIO.balance_weights, z, self.b)
        self.assertRaises(ValueError, rootIO.balance_weights, self.b, z)
        self.assertRaises(ValueError, rootIO.balance_weights, z, z)

    def test_raises_on_empty(self):
        """
        Check that ValueError is raised if either argument is empty.
        """
        e = np.array([])
        self.assertRaises(ValueError, rootIO.balance_weights, e, self.b)
        self.assertRaises(ValueError, rootIO.balance_weights, self.b, e)
        self.assertRaises(ValueError, rootIO.balance_weights, e, e)

    def test_raises_on_negative(self):
        """
        Check that ValueError is raised if normalisation is negative for one
        argument.
        """
        n = self.a - 1
        self.assertRaises(ValueError, rootIO.balance_weights, n, self.b)
        self.assertRaises(ValueError, rootIO.balance_weights, self.b, n)


class ReweightTests(unittest.TestCase):
    """
    Tests for rootIO.reweight
    """

    def setUp(self):
        self.a = np.random.rand(1000) - 0.25

    def test_normalisation_is_preserved(self):
        """
        Test that the normalisation is preserved.
        """
        self.assertTrue(np.isclose(self.a.sum(),
                                   rootIO.reweight(self.a.sum())))

    def test_all_weights_are_positive(self):
        """
        Test that all resulting weights are positive
        """
        self.assertTrue((rootIO.reweight(self.a) >= 0).all())

    def test_raises_on_negative_normalisation(self):
        """
        Test a ValueError is raised if the normalisation is negative.
        """
        self.assertRaises(ValueError, rootIO.reweight, self.a - 0.75)

    def test_raises_on_zero_normalisation(self):
        """
        Test a ValueError is raised if the normalisation is zero.
        """
        z = np.zeros(1000)
        az = np.append(self.a, -self.a)
        self.assertRaises(ValueError, rootIO.reweight, z)
        self.assertRaises(ValueError, rootIO.reweight, az)

    def test_raises_on_empty(self):
        """
        Test a ValueError is raised if the provided array is empty.
        """
        self.assertRaises(ValueError, rootIO.reweight, np.array([]))


class TH1NameFormatTests(unittest.TestCase):
    """
    Tests for rootIO._format_TH1_name
    """

    def setUp(self):
        self.name = "Ttree_PROCESS"
        self.name_up = "Ttree_PROCESS__SYSTEMATIC__plus"
        self.name_down = "Ttree_PROCESS__SYSTEMATIC__minus"

    def test_rename(self):
        """
        Test non-systematic histograms are renamed correctly.
        """
        self.assertEqual("MVA_all__PROCESS",
                         rootIO._format_TH1_name(self.name))

    def test_rename_systematic_up_combine(self):
        """
        Test systematic histograms for 1σ variations up are renamed correctly
        for combine.
        """
        self.assertEqual("MVA_all__PROCESS__SYSTEMATICUp",
                         rootIO._format_TH1_name(self.name_up))

    def test_rename_systematic_up_combine(self):
        """
        Test systematic histograms for 1σ variations down are renamed correctly
        for combine.
        """
        self.assertEqual("MVA_all__PROCESS__SYSTEMATICDown",
                         rootIO._format_TH1_name(self.name_down))

    def test_rename_systematic_up_THETA(self):
        """
        Test systematic histograms for 1σ variations up are renamed correctly
        for THETA.
        """
        self.assertEqual("MVA_all__PROCESS__SYSTEMATIC__plus",
                         rootIO._format_TH1_name(self.name_up, combine=False))

    def test_rename_systematic_up_THETA(self):
        """
        Test systematic histograms for 1σ variations down are renamed correctly
        for THETA.
        """
        self.assertEqual("MVA_all__PROCESS__SYSTEMATIC__minus",
                         rootIO._format_TH1_name(self.name_down,
                                                 combine=False))

    def test_raises_on_bad_name(self):
        """
        Test a ValueError is raised if the name is unchanged.
        """
        bad_name = "waerstftyj"
        self.assertRaises(ValueError, rootIO._format_TH1_name, bad_name)
        self.assertRaises(ValueError, rootIO._format_TH1_name, "")


class ColToTH1Tests(unittest.TestCase):
    """
    Tests for rootIO.col_to_TH1
    """

    def setUp(self):
        self.a = np.random.rand(1000)
        self.w = np.random.rand(1000) - 0.25

    def test_returns_TH1D_without_weights(self):
        """
        Ensures a TH1D is returned when no weights are provided.
        """
        self.assertIsInstance(rootIO.col_to_TH1(self.a), ROOT.TH1D)

    def test_returns_TH1D_with_weights(self):
        """
        Ensures a TH1D is returned when weights are provided.
        """
        self.assertIsInstance(rootIO.col_to_TH1(self.a, w=self.w), ROOT.TH1D)

    def test_bin_totals_preserved_without_weights(self):
        """
        Check the bin totals are unchanged when no weights are provided.
        """
        bins = 20
        bin_range = (0, 1)
        hist, _ = np.histogram(self.a, bins=bins, range=bin_range)
        h = rootIO.col_to_TH1(self.a, bins=bins, range=bin_range)
        for i in range(0, bins):
            self.assertEqual(hist[i], h.GetBinContent(i + 1))

    def test_bin_totals_preserved_with_weights(self):
        """
        Check the bin totals are unchanged when weights are provided.
        """
        bins = 20
        bin_range = (0, 1)
        hist, _ = np.histogram(self.a, bins=bins, weights=self.w,
                               range=bin_range)
        h = rootIO.col_to_TH1(self.a, bins=bins, w=self.w, range=bin_range)
        for i in range(0, bins):
            self.assertEqual(hist[i], h.GetBinContent(i + 1))

    def test_poisson_weights(self):
        """
        Check errors are poissonian if entries are unweighted.
        """
        bins = 20
        bin_range = (0, 1)
        hist, _ = np.histogram(self.a, bins=bins, range=bin_range)
        h = rootIO.col_to_TH1(self.a, bins=bins, range=bin_range)
        for i in range(0, bins):
            self.assertEqual(np.sqrt(hist[i]), h.GetBinError(i + 1))

    def test_sum_square_weights(self):
        """
        Check errors are the square root of the sum of the square weights if
        entries are weighted.
        """
        bins = 20
        bin_range = (0, 1)
        histerror, _ = np.histogram(self.a, bins=bins, weights=self.w ** 2,
                                    range=bin_range)
        h = rootIO.col_to_TH1(self.a, bins=bins, w=self.w, range=bin_range)
        for i in range(0, bins):
            self.assertEqual(np.sqrt(histerror[i]), h.GetBinError(i + 1))

    def test_empty(self):
        """
        Test if an empty histogram is returned with empty input.
        """
        bins = 20
        h = rootIO.col_to_TH1(np.array([]), bins=bins)
        self.assertIsInstance(h, ROOT.TH1D)
        self.assertEqual(sum(h.GetBinContent(i) for i in range(1, bins + 1)),
                         0)


class PoissonPseudodataTests(unittest.TestCase):
    """
    Tests for rootIO.poisson_pseudodata
    """

    # TODO: There should be some sort of test on the bin contents

    def setUp(self):
        self.a = np.random.rand(1000) - 0.5

    def test_TH1_returned(self):
        """
        Check a TH1D is returned.
        """
        self.assertIsInstance(rootIO.poisson_pseudodata(self.a), ROOT.TH1D)

    def test_same_bin_number(self):
        """
        Check the number of bins is unchanged.
        """
        bins = 40
        self.assertEqual(rootIO.poisson_pseudodata(self.a,
                                                   bins=bins).GetNbinsX(),
                         bins)

    def test_empty(self):
        """
        Check an empty input returns an empty histogram.
        """
        bins = 20
        h = rootIO.poisson_pseudodata(np.array([]), bins=20)
        self.assertEqual(sum(h.GetBinContent(i) for i in range(1, bins + 1)),
                         0)


if __name__ == "__main__":
    unittest.main()
