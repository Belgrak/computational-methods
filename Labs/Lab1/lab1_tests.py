import unittest
import numpy as np
from numpy.testing import assert_allclose
from condition_numbers import calculate_spectral_number, calculate_volume_number, calculate_angular_number
from math import sqrt

matrix = np.array([[1, 2],
                   [3, 4]])


# inv = np.array([-2, 1],
#                [1.5, -0.5])


class Lab1TestCase(unittest.TestCase):
    def test_spectral_num(self):
        assert_allclose(round(calculate_spectral_number(matrix), 2), 15)

    def test_volume_num(self):
        assert_allclose(calculate_volume_number(matrix), 5 * sqrt(5) / 2)

    def test_angular_num(self):
        assert_allclose(calculate_angular_number(matrix), sqrt(31.25))


if __name__ == '__main__':
    unittest.main()
