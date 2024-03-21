from numpy.linalg import norm, inv, det
from functools import reduce
from math import sqrt


def calculate_spectral_number(a):
    return norm(a) * norm(inv(a))


def calculate_volume_number(a):
    return reduce(lambda x, y: x * y, [sqrt(sum(a[n][m] ** 2 for m in range(a.shape[0]))) for n in range(a.shape[0])],
                  1) / abs(det(a))


def calculate_angular_number(a):
    a_inv = inv(a)
    return max(norm(a[i, :]) * norm(a_inv[:, i]) for i in range(a.shape[0]))
