import numpy as np
from numpy.linalg import cond, solve, norm
from tabulate import tabulate

from matrices import generate_hilbert_matrix, generate_tridiagonal_matrix, pakulina_matrix
from condition_numbers import calculate_spectral_number, calculate_volume_number, calculate_angular_number


def generate_right_hand_side(n):
    return np.random.rand(n)


def pakulina_right_hand_side(n):
    return np.array([7.570463, 8.876384, 3.411906])


def calculate_error(x, x_tilde):
    return norm(x - x_tilde)


# Функция для выполнения эксперимента
def run_experiment(matrix_generator, rhs_generator, size_range):
    results = []
    for n in size_range:
        A = matrix_generator(n)
        b = rhs_generator(n)
        x = solve(A, b)

        spec_num = calculate_spectral_number(A)
        vol_num = calculate_volume_number(A)
        ang_num = calculate_angular_number(A)

        for epsilon in np.logspace(-2, -10, num=9):
            b_new = b + epsilon * np.random.rand(n)
            x_tilde = solve(A, b_new)
            error = calculate_error(x, x_tilde)
            results.append([f"{n:.20f}", f"{spec_num:.20f}", f"{vol_num:.20f}", f"{ang_num:.20f}", f"{epsilon:.20f}",
                            f"{error:.20f}"])
    headers = ["Matrix size", "Spectral num", "Volume num", "Angular num", "Epsilon", "Error"]
    return tabulate(results, headers=headers, tablefmt="grid")


# Запуск эксперимента с матрицами Гильберта
print("Experiments with Hilbert matrices:")
print(run_experiment(generate_hilbert_matrix, generate_right_hand_side, range(3, 11)))

# Запуск эксперимента с трехдиагональными матрицами
print("\nExperiments with tridiagonal matrices:")
print(run_experiment(lambda n: generate_tridiagonal_matrix(n, 2, 1), generate_right_hand_side, range(3, 11)))

# Запуск эксперимента с матрицей из 7 варианта
print("\nExperiments with matrix from book:")
print(run_experiment(pakulina_matrix, pakulina_right_hand_side, range(3, 4)))
