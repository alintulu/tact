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
    Tests for tact.rootIO.balance_weights
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
        Check that ValueError is raised if normalisation is negative.
        """
        n = self.a - 1
        self.assertRaises(ValueError, rootIO.balance_weights, n, self.b)
        self.assertRaises(ValueError, rootIO.balance_weights, self.b, n)


if __name__ == '__main__':
    unittest.main()
