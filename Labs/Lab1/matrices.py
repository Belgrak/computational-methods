import numpy as np


def generate_hilbert_matrix(n):
    return np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])


def generate_tridiagonal_matrix(n, diagonal_value, off_diagonal_value):
    main_diag = np.full(n, diagonal_value)
    off_diag = np.full(n - 1, off_diagonal_value)
    return np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)


def pakulina_matrix(n):
    return np.array([[9.331343, 1.120045, -2.880925],
                     [1.120045, 7.086042, 0.670297],
                     [-2.880925, 0.670297, 5.622534]])
