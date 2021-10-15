import warnings

import numpy as np


def power_to_infinity(matrix):
    """Raises a square matrix to an infinite power using eigendecomposition.

    All matrix rows must add to 1.
    M = Q*L*inv(Q), where L = eigenvalue diagonal values, Q = eigenvector matrix
    M^N = Q*(L^N)*inv(Q)
    """
    eigen_value, eigen_vectors = np.linalg.eig(matrix)

    # At infinity everything converges to 0 or 1, thus we use np.trunc()
    diagonal = np.diag(np.trunc(eigen_value.real + 0.001))
    try:
        result = (eigen_vectors @ diagonal @ np.linalg.inv(eigen_vectors)).real
    except np.linalg.LinAlgError as err:
        if "Singular matrix" in str(err):
            warnings.warn(
                "Warning... Singular matrix error. Check for lines or cols "
                + "fully filled with zeros."
            )
            result = (eigen_vectors @ diagonal @ np.linalg.pinv(eigen_vectors)).real
        else:
            raise
    return result


def normalize_rows(matrix):
    size = matrix.shape[0]
    mean = matrix.sum(axis=1).reshape((size, 1))
    mean = np.where(mean == 0, 1, mean)
    return matrix / mean


def calc_total_conversion(matrix):
    normal_matrix = normalize_rows(matrix)
    infinity_matrix = power_to_infinity(normal_matrix)
    return infinity_matrix[0, -1]


def removal_effect(matrix):
    size = matrix.shape[0]
    conversions = np.zeros(size)
    for column in range(1, size - 2):
        temp = matrix.copy()
        temp[:, -2] = temp[:, -2] + temp[:, column]
        temp[:, column] = 0
        conversions[column] = calc_total_conversion(temp)
    conversion_orig = calc_total_conversion(matrix)
    return 1 - (conversions / conversion_orig)


def path_to_matrix(paths):
    channel_max = int(paths[:, 0:2].max()) + 1
    matrix = np.zeros((channel_max, channel_max), dtype="float")
    for x, y, val in paths:
        matrix[int(x), int(y)] = val
    matrix[-1, -1] = 1
    matrix[-2, -2] = 1
    return matrix


def save_orig_dest(arr):
    orig = []
    dest = []
    journey_length = []
    orig.extend(arr[:-1])
    dest.extend(arr[1:])
    journey_length.append(len(arr))
