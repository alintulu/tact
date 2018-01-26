# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
