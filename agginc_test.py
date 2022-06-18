"""
This file contains our three tests:
MMDAggInc, HSICAggInc and KSDAggInc
which are implemented in the function agginc()
"""

import numpy as np
import itertools
import scipy.spatial


def create_indices(N, R):
    """
    Return lists of indices of R subdiagonals of N x N matrix
    """
    index_X = list(
        itertools.chain(*[[i for i in range(N - r)] for r in range(1, R + 1)])
    )
    index_Y = list(
        itertools.chain(*[[i + r for i in range(N - r)] for r in range(1, R + 1)])
    )
    return index_X, index_Y


def compute_h_MMD_values(X, Y, R, bandwidths, return_indices_N=False):
    """
    Compute h_MMD values.

    inputs:
        X (m,d)
        Y (n,d)
        R int
        bandwidths (#bandwidths,)

    output (#bandwidths, R * N - R * (R - 1) / 2)
    """
    N = min(X.shape[0], Y.shape[0])
    assert X.shape[1] == Y.shape[1]

    index_i, index_j = create_indices(N, R)

    norm_Xi_Xj = np.linalg.norm(X[index_i] - X[index_j], axis=1) ** 2
    norm_Xi_Yj = np.linalg.norm(X[index_i] - Y[index_j], axis=1) ** 2
    norm_Yi_Xj = np.linalg.norm(Y[index_i] - X[index_j], axis=1) ** 2
    norm_Yi_Yj = np.linalg.norm(Y[index_i] - Y[index_j], axis=1) ** 2

    h_values = np.zeros((bandwidths.shape[0], norm_Xi_Xj.shape[0]))
    for r in range(bandwidths.shape[0]):
        K_Xi_Xj_b = np.exp(-norm_Xi_Xj / bandwidths[r] ** 2)
        K_Xi_Yj_b = np.exp(-norm_Xi_Yj / bandwidths[r] ** 2)
        K_Yi_Xj_b = np.exp(-norm_Yi_Xj / bandwidths[r] ** 2)
        K_Yi_Yj_b = np.exp(-norm_Yi_Yj / bandwidths[r] ** 2)
        h_values[r] = K_Xi_Xj_b - K_Xi_Yj_b - K_Yi_Xj_b + K_Yi_Yj_b

    if return_indices_N:
        return h_values, index_i, index_j, N
    else:
        return h_values


def compute_h_HSIC_values(X, Y, R, bandwidths, return_indices_N=False):
    """
    Compute h_HSIC values.

    inputs:
        X (m,d)
        Y (m,d)
        R int
        bandwidths (2, #bandwidths) (bandwidths for X and for Y)

    output (#bandwidths, R * N - R * (R - 1) / 2)
    """
    assert X.shape[0] == Y.shape[0]
    N = int(X.shape[0] / 2)

    bandwidths_X = np.array(bandwidths[0])
    bandwidths_Y = np.array(bandwidths[1])

    h_X_values, index_i, index_j, Nbis = compute_h_MMD_values(
        X[:N], X[N:], R, bandwidths_X, True
    )
    assert N == Nbis
    h_Y_values = compute_h_MMD_values(Y[:N], Y[N:], R, bandwidths_Y)

    # we need to consider all pairs of bandwidths
    h_XY_values = (
        np.expand_dims(h_X_values, 0) * np.expand_dims(h_Y_values, 1)
    ).reshape(h_X_values.shape[0] * h_Y_values.shape[0], -1)

    if return_indices_N:
        return h_XY_values, index_i, index_j, N
    else:
        return h_XY_values


def compute_h_KSD_values(
    X, score_X, R, bandwidths, return_indices_N=False, beta_imq=-0.5
):
    """
    Compute h_KSD values.

    inputs:
        X (m,d)
        Y (m,d)
        R int
        bandwidths (#bandwidths,)
        beta_imq in (-1, 0)

    output (#bandwidths, R * N - R * (R - 1) / 2)
    """
    assert X.shape == score_X.shape
    N, d = X.shape

    index_i, index_j = create_indices(N, R)

    Xi_minus_Xj = X[index_i] - X[index_j]
    norm_Xi_Xj = np.linalg.norm(Xi_minus_Xj, axis=1) ** 2
    sXi = score_X[index_i]
    sXj = score_X[index_j]
    sXi_minus_sXj = sXi - sXj
    sXi_minus_sXj_dot_Xi_minus_Xj = np.einsum(
        "ij,ij->i", sXi_minus_sXj, Xi_minus_Xj, optimize=True
    )
    sXi_dot_sXj = np.einsum("ij,ij->i", sXi, sXj, optimize=True)

    h_values = np.zeros((bandwidths.shape[0], Xi_minus_Xj.shape[0]))
    for r in range(bandwidths.shape[0]):
        b_norm_Xi_Xj = bandwidths[r] ** 2 + norm_Xi_Xj
        h_values[r] = (
            sXi_dot_sXj * b_norm_Xi_Xj**beta_imq
            - sXi_minus_sXj_dot_Xi_minus_Xj
            * 2
            * beta_imq
            * b_norm_Xi_Xj ** (beta_imq - 1)
            - 2 * d * beta_imq * b_norm_Xi_Xj ** (beta_imq - 1)
            - 4
            * beta_imq
            * (beta_imq - 1)
            * norm_Xi_Xj
            * b_norm_Xi_Xj ** (beta_imq - 2)
        )

    if return_indices_N:
        return h_values, index_i, index_j, N
    else:
        return h_values


def compute_bootstrap_values(
    h_values, index_i, index_j, N, B, seed, return_original=False
):
    """
    Compute B bootstrap values.

    inputs:
        h_values, index_i, index_j = compute_h_XXX_values(...)
        h_values (#bandwidths, R * N - R * (R - 1) / 2)
        N int
        B int
        seed int


    output (#bandwidths, B)
    """
    rs = np.random.RandomState(seed)
    epsilon = rs.choice([1.0, -1.0], size=(N, B))
    e_values = epsilon[index_i] * epsilon[index_j]
    bootstrap_values = h_values @ e_values
    if return_original:
        original_value = h_values @ np.ones(h_values.shape[1])
        return bootstrap_values, original_value
    else:
        return bootstrap_values


def compute_quantile(bootstrap_values, original_value, B1, B2, B3, alpha, weights_type):
    """
    Compute quantile.

    inputs:
        bootstrap_values (#bandwidths, B1 + B2)
        original_value (#bandwidths, )
        B1 int
        B2 int
        B3 int
        alpha in (0,1)
        weights_type "uniform" or "decreasing" or "increasing" or "centred"

    returns quantiles (#bandwidths, )
    """
    # this can be adjusted B1 for quantiles, B2 for P_u
    bootstrap_1 = np.column_stack([bootstrap_values[:, :B1], original_value])
    bootstrap_1_sorted = np.sort(bootstrap_1)  # sort each row
    bootstrap_2 = bootstrap_values[:, B1:]
    assert B2 == bootstrap_2.shape[1]

    weights = create_weights(bootstrap_values.shape[0], weights_type)
    # (1-u*w_lambda)-quantiles for the #bandwidths
    quantiles = np.zeros((bootstrap_values.shape[0], 1))
    u_min = 0
    u_max = np.min(1 / weights)
    for _ in range(B3):
        u = (u_max + u_min) / 2
        for i in range(bootstrap_values.shape[0]):
            quantiles[i] = bootstrap_1_sorted[
                i, int(np.ceil((B1 + 1) * (1 - u * weights[i]))) - 1
            ]
        P_u = np.sum(np.max(bootstrap_2 - quantiles, 0) > 0) / B2
        if P_u <= alpha:
            u_min = u
        else:
            u_max = u
    u = u_min
    for i in range(bootstrap_values.shape[0]):
        quantiles[i] = bootstrap_1_sorted[
            i, int(np.ceil((B1 + 1) * (1 - u * weights[i]))) - 1
        ]
    return quantiles


def agginc(agg_type, X, Y, alpha, R, l_minus, l_plus, B1, B2, B3, weights_type, seed):
    """
    Efficient Aggregated Kernel Tests
    AggInc tests: MMDAggInc, HSICAggInc and KSDAggInc

    inputs:
        agg_type in "mmd", "hsic" or "ksd" (Gaussian kernel for "mmd" and "hsic", IMQ kernel for "ksd")
        X (m, d)
        Y (n, d) (for ksd Y is score_X and n = m, for hsic n = m)
        alpha in (0, 1)
        R int
        l_minus, l_plus int (powers of 2 for the collection of bandwidths)
            for hsic l_minus and l_plus are each pairs
        B1, B2, B3 int
        weights_type "uniform" or "decreasing" or "increasing" or "centred"
        seed int

    output: 0 (fail to reject H_0) or 1 (reject H_0)
    """
    if agg_type == "mmd":
        compute_h_values = compute_h_MMD_values
        Z = np.concatenate((X, Y))
        median_bandwidth = compute_median_bandwidth(seed, Z)
        bandwidths = np.array(
            [2**i * median_bandwidth for i in range(l_minus, l_plus + 1)]
        )
    elif agg_type == "hsic":
        assert weights_type == "uniform"
        assert type(l_minus) == type(l_plus)
        if type(l_minus) == int:
            l_minus = (l_minus, l_minus)
            l_plus = (l_plus, l_plus)
        elif len(np.array(l_minus)) > 2:
            raise ValueError(
                "l_minus and l_plus should either be (signed) integers or pairs."
            )
        compute_h_values = compute_h_HSIC_values
        median_bandwidth_X = compute_median_bandwidth(seed, X)
        median_bandwidth_Y = compute_median_bandwidth(seed + 1, Y)
        median_bandwidths = (median_bandwidth_X, median_bandwidth_Y)
        bandwidths = [
            [2**i * median_bandwidths[j] for i in range(l_minus[j], l_plus[j] + 1)]
            for j in range(2)
        ]
    elif agg_type == "ksd":
        compute_h_values = compute_h_KSD_values
        median_bandwidth = compute_median_bandwidth(seed, X)
        bandwidths = np.array(
            [2**i * median_bandwidth for i in range(l_minus, l_plus + 1)]
        )
    else:
        raise ValueError('The value of agg_type should be "mmd" or' '"hsic" or "ksd".')

    h_values, index_i, index_j, N = compute_h_values(
        X, Y, R, bandwidths, return_indices_N=True
    )
    bootstrap_values, original_value = compute_bootstrap_values(
        h_values, index_i, index_j, N, B1 + B2, seed, return_original=True
    )
    quantiles = compute_quantile(
        bootstrap_values, original_value, B1, B2, B3, alpha, weights_type
    )

    for i in range(original_value.shape[0]):
        if original_value[i] > quantiles[i]:
            return 1
    return 0


def create_weights(N, weights_type):
    """
    Create weights as defined in Section 5.1 of MMD Aggregated Two-Sample Test (Schrab et al., 2021).
    inputs: N: number of bandwidths to test
            weights_type: "uniform" or "decreasing" or "increasing" or "centred"
    output: (N,) array of weights
    """
    if weights_type == "uniform":
        weights = np.array(
            [
                1 / N,
            ]
            * N
        )
    elif weights_type == "decreasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = np.array([1 / (i * normaliser) for i in range(1, N + 1)])
    elif weights_type == "increasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = np.array([1 / ((N + 1 - i) * normaliser) for i in range(1, N + 1)])
    elif weights_type == "centred":
        if N % 2 == 1:
            normaliser = sum([1 / (abs((N + 1) / 2 - i) + 1) for i in range(1, N + 1)])
            weights = np.array(
                [1 / ((abs((N + 1) / 2 - i) + 1) * normaliser) for i in range(1, N + 1)]
            )
        else:
            normaliser = sum(
                [1 / (abs((N + 1) / 2 - i) + 0.5) for i in range(1, N + 1)]
            )
            weights = np.array(
                [
                    1 / ((abs((N + 1) / 2 - i) + 0.5) * normaliser)
                    for i in range(1, N + 1)
                ]
            )
    else:
        raise ValueError(
            'The value of weights_type should be "uniform" or'
            '"decreasing" or "increasing" or "centred".'
        )
    return weights


def compute_median_bandwidth(seed, Z, max_samples=500, min_value=0.0001, shuffle=False):
    """
    Compute the median L^2-distance between all the points in Z using at
    most max_samples samples and using a minimum threshold value min_value.
    inputs: seed: non-negative integer
            X: (m,d) array of samples
            max_samples: number of samples used to compute the median (int or None)
    output: median bandwidth (float)
    """
    if max_samples != None:
        if shuffle:
            rs = np.random.RandomState(seed)
            pZ = rs.choice(X.shape[0], min(max_samples, Z.shape[0]), replace=False)
            median_bandwidth = np.median(
                scipy.spatial.distance.pdist(Z[pZ], "euclidean")
            )
        else:
            median_bandwidth = np.median(
                scipy.spatial.distance.pdist(
                    Z[: min(max_samples, Z.shape[0])], "euclidean"
                )
            )
    else:
        median_bandwidth = np.median(scipy.spatial.distance.pdist(Z, "euclidean"))
    return np.maximum(median_bandwidth, min_value)


def inc_median(type, X, Y, alpha, R, B, seed):
    """
    Efficient test using median bandwidth (no aggregation)

    inputs:
        type in "mmd", "hsic" or "ksd" (Gaussian kernel for "mmd" and "hsic", IMQ kernel for "ksd")
        X (m, d)
        Y (n, d) (for ksd Y is score_X and n = m, for hsic n = m)
        alpha in (0, 1)
        R int
        B int
        seed int

    output: 0 (fail to reject H_0) or 1 (reject H_0)
    """
    if agg_type == "mmd":
        compute_h_values = compute_h_MMD_values
        Z = np.concatenate((X, Y))
        median_bandwidth = compute_median_bandwidth(seed, Z)
        bandwidths = np.array([median_bandwidth])
    elif agg_type == "hsic":
        compute_h_values = compute_h_HSIC_values
        median_bandwidth_X = compute_median_bandwidth(seed, X)
        median_bandwidth_Y = compute_median_bandwidth(seed + 1, Y)
        bandwidths = (np.array([median_bandwidth_X]), np.array([median_bandwidth_Y]))
    elif agg_type == "ksd":
        compute_h_values = compute_h_KSD_values
        median_bandwidth = compute_median_bandwidth(seed, X)
        bandwidths = bandwidths = np.array([median_bandwidth])
    else:
        raise ValueError('The value of agg_type should be "mmd" or' '"hsic" or "ksd".')

    h_values, index_i, index_j, N = compute_h_values(
        X, Y, R, bandwidths, return_indices_N=True
    )
    bootstrap_values, original_value = compute_bootstrap_values(
        h_values, index_i, index_j, N, B, seed, return_original=True
    )
    assert bootstrap_values.shape[0] == 1

    bootstrap_1 = np.column_stack([bootstrap_values, original_value])
    bootstrap_1_sorted = np.sort(bootstrap_1)
    quantile = bootstrap_1_sorted[0, int(np.ceil((B + 1) * (1 - alpha))) - 1]
    if original_value > quantile:
        return 1
    return 0
